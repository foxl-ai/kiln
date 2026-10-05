#!/usr/bin/env bash
# Kiln dev/bench fleet on EC2 spot. Every resource this creates carries Project=kiln,
# and `down` only ever touches instances carrying that tag.
#
#   infra/fleet.sh bootstrap                  # one-time: IAM role + instance profile + SG
#   infra/fleet.sh up <name> <type> [az]      # launch a spot instance (SSM only, no SSH);
#                                             # KILN_SPOT_MAX_PRICE=<$/h> refuses a pricier AZ,
#                                             # KILN_DRY_RUN=1 checks the request, launches nothing;
#                                             # KILN_MARKET=ondemand|capacity-block (see cmd_up)
#   infra/fleet.sh ready <name>               # wait out the account's SSM patch reboot
#   infra/fleet.sh ls                         # list Project=kiln instances
#   infra/fleet.sh run <name> '<shell>'       # run a command via SSM, print output
#   infra/fleet.sh sync <name>                # ship this checkout to /opt/kiln/src on the box
#   infra/fleet.sh py <name> <file.py> [args] # sync, then run a script from this checkout
#   infra/fleet.sh bg <name> <file.py> [args] # sync, then start the script detached; prints its log
#   infra/fleet.sh log <name> <log> [lines]   # tail a log on the box (and say if the job still runs)
#   infra/fleet.sh nvme <name>                # RAID0 the instance-store NVMe at /opt/kiln/nvme (+ HF cache)
#
# "This checkout" is the one this script lives in, wherever it is called from.
# KILN_BOX_SRC=/opt/kiln/src-<x> syncs to (and runs from) another directory on the box, so a
# second checkout can run CPU-only work beside a job that is running from /opt/kiln/src.
#   infra/fleet.sh down <name>|--all          # terminate
#   infra/fleet.sh ls --all                   # Project=kiln instances in every Kiln region
#
# Region: us-east-2 by default (measured 2026-10-02: the only region offering trn1, trn1n,
# trn2 and inf2 together, and the cheapest spot for each - trn1.32xlarge $2.15/h,
# trn2.48xlarge $14.29/h, inf2.48xlarge $1.30/h, trn1.2xlarge $0.134/h). Any other region
# with `--region <r>` before the subcommand, or KILN_REGION=<r>: trn2.3xlarge exists only in
# sa-east-1, ap-southeast-4 and ap-south-2 (docs/research/trn2-trn3.md 1.1). Run `bootstrap`
# once per region (it makes that region's security group; the IAM role is global). The S3
# bucket stays in us-east-2 and every region reads and writes it cross-region.
set -euo pipefail

if [ "${1:-}" = "--region" ]; then KILN_REGION="$2"; shift 2; fi

export AWS_CONFIG_FILE="${AWS_CONFIG_FILE:-$HOME/.aws/config}"
export AWS_SHARED_CREDENTIALS_FILE="${AWS_SHARED_CREDENTIALS_FILE:-$HOME/.aws/credentials}"
export AWS_PROFILE="${AWS_PROFILE:-default}"
REGION="${KILN_REGION:-us-east-2}"
# Regions `ls --all` scans: us-east-2 plus the three that offer trn2.3xlarge.
KILN_REGIONS="us-east-2 sa-east-1 ap-southeast-4 ap-south-2"
ROLE=kiln-ec2
SG_NAME=kiln-ssm
# Deep Learning AMI Neuron (Ubuntu 24.04), multi-framework. Resolved at launch time from
# AWS's own SSM parameter IN THE LAUNCH REGION so the SDK tracks the latest DLAMI (2.32.0 on
# 2026-08-18: ami-0222021b369f03219 in us-east-2, ami-06f5c32a48089ad7a in sa-east-1,
# ami-01f66e576e60931ed in ap-southeast-4, read 2026-10-03). ap-south-2 has no such
# parameter ("not a valid namespace"); set KILN_AMI to an AMI id there.
AMI_PARAM="${KILN_AMI_PARAM:-/aws/service/neuron/dlami/multi-framework/ubuntu-24.04/latest/image_id}"
AMI="${KILN_AMI:-}"
ROOT_GB="${KILN_ROOT_GB:-300}"
TAGS="Key=Project,Value=kiln"
ACCOUNT="${KILN_AWS_ACCOUNT:-$(aws sts get-caller-identity --query Account --output text)}"
BUCKET="${KILN_BUCKET:-kiln-$ACCOUNT-use2}"
BUCKET_REGION=us-east-2  # where $BUCKET lives, whatever region the instance runs in
# The venv vllm-neuron ships in on the SDK 2.32 DLAMI. Kiln runs in it because it holds
# the matched torch / torch-xla / neuronx-cc / nki builds for this SDK.
VENV="${KILN_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0}"
SRC="${KILN_BOX_SRC:-/opt/kiln/src}"

aws_() { aws --region "$REGION" "$@"; }
s3_() { aws --region "$BUCKET_REGION" s3 "$@"; }

sg_id() {
  aws_ ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" "Name=tag:Project,Values=kiln" \
    --query 'SecurityGroups[0].GroupId' --output text
}

instance_id() {
  aws_ ec2 describe-instances \
    --filters "Name=tag:Project,Values=kiln" "Name=tag:Name,Values=$1" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text
}

cmd_bootstrap() {
  if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
    aws iam create-role --role-name "$ROLE" --tags "$TAGS" \
      --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
    aws iam attach-role-policy --role-name "$ROLE" \
      --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
    aws iam create-instance-profile --instance-profile-name "$ROLE" --tags "$TAGS" >/dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
    echo "created role + instance profile $ROLE"
  else
    echo "role $ROLE exists"
  fi
  local s3api="aws --region $BUCKET_REGION s3api"
  if ! $s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; then
    $s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint="$BUCKET_REGION" >/dev/null
    $s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
      BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
    $s3api put-bucket-tagging --bucket "$BUCKET" --tagging "TagSet=[{$TAGS}]"
    echo "created bucket $BUCKET"
  fi
  aws iam put-role-policy --role-name "$ROLE" --policy-name kiln-bucket --policy-document \
    "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:ListBucket\",\"s3:DeleteObject\"],\"Resource\":[\"arn:aws:s3:::$BUCKET\",\"arn:aws:s3:::$BUCKET/*\"]}]}"
  if [ "$(sg_id)" = "None" ]; then
    vpc=$(aws_ ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
    sg=$(aws_ ec2 create-security-group --group-name "$SG_NAME" --vpc-id "$vpc" \
      --description "Kiln fleet: no inbound, SSM only" \
      --tag-specifications "ResourceType=security-group,Tags=[{$TAGS}]" --query GroupId --output text)
    echo "created security group $sg in $REGION (no inbound rules)"
  else
    echo "security group $(sg_id) exists in $REGION"
  fi
}

cmd_up() {
  local name="$1" type="$2" az="${3:-}"
  local existing; existing=$(instance_id "$name")
  if [ "$existing" != "None" ]; then echo "$name already exists: $existing"; return; fi
  local sg; sg=$(sg_id)
  [ "$sg" = "None" ] && { echo "no $SG_NAME security group in $REGION: run 'infra/fleet.sh --region $REGION bootstrap'" >&2; exit 1; }
  local ami="$AMI"
  [ -z "$ami" ] && ami=$(aws_ ssm get-parameter --name "$AMI_PARAM" --query Parameter.Value --output text)
  local placement=()
  if [ -n "$az" ]; then
    local subnet; subnet=$(aws_ ec2 describe-subnets --filters "Name=availability-zone,Values=$az" "Name=default-for-az,Values=true" --query 'Subnets[0].SubnetId' --output text)
    placement=(--subnet-id "$subnet")
  fi
  # KILN_MARKET: spot (default), ondemand, or capacity-block with KILN_CAPACITY_RESERVATION=cr-...
  # (an EC2 Capacity Block for ML: MarketType=capacity-block plus the reservation as target).
  local market=()
  case "${KILN_MARKET:-spot}" in
    spot) market=(--instance-market-options "MarketType=spot,SpotOptions={${KILN_SPOT_MAX_PRICE:+MaxPrice=$KILN_SPOT_MAX_PRICE,}SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}") ;;
    ondemand) ;;  # no market option = on-demand
    capacity-block) market=(--instance-market-options MarketType=capacity-block
      --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=${KILN_CAPACITY_RESERVATION:?KILN_CAPACITY_RESERVATION=cr-...}}") ;;
    *) echo "KILN_MARKET must be spot, ondemand or capacity-block" >&2; exit 2 ;;
  esac
  aws_ ec2 run-instances --image-id "$ami" --instance-type "$type" \
    --iam-instance-profile "Name=$ROLE" --security-group-ids "$sg" \
    ${placement[@]+"${placement[@]}"} ${KILN_DRY_RUN:+--dry-run} \
    ${market[@]+"${market[@]}"} \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$ROOT_GB,VolumeType=gp3,Iops=6000,Throughput=500,DeleteOnTermination=true}" \
    --metadata-options HttpTokens=required \
    --tag-specifications "ResourceType=instance,Tags=[{$TAGS},{Key=Name,Value=$name}]" \
                         "ResourceType=volume,Tags=[{$TAGS},{Key=Name,Value=$name}]" \
    --query 'Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone]' --output text
  echo "ami $ami (${KILN_AMI:+KILN_AMI}${KILN_AMI:-$AMI_PARAM}) region $REGION"
}

cmd_ready() {
  # The account carries an SSM association (<aws-account-id>-PatchWeekly: AWS-RunPatchBaseline,
  # Operation=Install, targets InstanceIds *, RebootOption unset = RebootIfNeeded) that
  # patches every instance when its agent registers and then REBOOTS it, a few minutes
  # after launch (measured: kiln-dev-trn1 patched 06:08-06:13 UTC, kiln-mimo-trn1
  # 10:56-11:03 then rebooted 11:04; 335 packages each). Kiln needs SSM, so instead of
  # disabling the agent (what pilot's CI runners do) wait until patching has finished AND
  # the machine has booted after it.
  local name="$1" id; id=$(instance_id "$name")
  # Not every region carries the association (sa-east-1 does, ap-southeast-4 did not on
  # 2026-10-03, `ssm list-associations`). Without it nothing reboots the box: wait for SSM only.
  local assoc; assoc=$(aws_ ssm list-associations --association-filter-list key=Name,value=AWS-RunPatchBaseline \
    --query 'Associations[0].Name' --output text 2>/dev/null || echo None)
  if [ "$assoc" = "None" ] || [ -z "$assoc" ]; then
    for _ in $(seq 1 60); do
      if cmd_run "$name" "true" 2>/dev/null | grep -q '^Success'; then
        echo "$name ready (no patch association in $REGION)"; return
      fi
      sleep 20
    done
    echo "$name: SSM not reachable after 20 min" >&2; return 1
  fi
  for _ in $(seq 1 90); do
    local end; end=$(aws_ ssm describe-instance-patch-states --instance-ids "$id" \
      --query 'InstancePatchStates[0].OperationEndTime' --output text 2>/dev/null || echo None)
    if [ "$end" != "None" ] && [ -n "$end" ]; then
      local boot; boot=$(cmd_run "$name" "date -u -d \"\$(uptime -s)\" +%s" 2>/dev/null | awk '/^Success/{print $2}')
      local endts; endts=$(python3 -c "import sys,datetime; print(int(datetime.datetime.fromisoformat(sys.argv[1]).timestamp()))" "$end")
      if [ -n "$boot" ] && [ "$boot" -gt "$endts" ]; then echo "$name ready (patched at $end, booted after)"; return; fi
    fi
    sleep 20
  done
  echo "$name: patch reboot not observed after 30 min" >&2; return 1
}

cmd_nvme() {
  # Stripe every instance-store NVMe disk (lsblk model "Amazon EC2 NVMe Instance Storage") as
  # RAID0, ext4, at /opt/kiln/nvme, and put the Hugging Face cache on it. Wiped on stop or
  # replacement and not in fstab: re-run after a reboot (the array is re-assembled, not rebuilt).
  cmd_run "$1" "set -e; mkdir -p /opt/kiln/nvme; if mountpoint -q /opt/kiln/nvme; then echo already mounted; else
    devs=\$(lsblk -dpno NAME,MODEL | awk '/Instance Storage/{print \$1}'); n=\$(echo \$devs | wc -w);
    [ \$n -gt 0 ] || { echo no instance-store disks; exit 1; };
    if mdadm --detail /dev/md0 >/dev/null 2>&1 || mdadm --assemble /dev/md0 \$devs 2>/dev/null; then mount /dev/md0 /opt/kiln/nvme;
    else mdadm --create /dev/md0 --level=0 --raid-devices=\$n \$devs --run; mkfs.ext4 -F -q -E lazy_itable_init=1,lazy_journal_init=1,nodiscard /dev/md0; mount /dev/md0 /opt/kiln/nvme; fi; fi;
    mkdir -p /opt/kiln/nvme/hf; if [ ! -L /opt/kiln/hf ]; then [ -d /opt/kiln/hf ] && cp -a /opt/kiln/hf/. /opt/kiln/nvme/hf/ && rm -rf /opt/kiln/hf; ln -sfn /opt/kiln/nvme/hf /opt/kiln/hf; fi;
    df -h /opt/kiln/nvme | tail -1"
}

cmd_ls() {
  if [ "${1:-}" = "--all" ]; then
    local r; for r in $KILN_REGIONS; do
      aws --region "$r" ec2 describe-instances --filters "Name=tag:Project,Values=kiln" \
        "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down" \
        --query 'Reservations[].Instances[].[Tags[?Key==`Name`]|[0].Value,InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime]' \
        --output text
    done
    return
  fi
  aws_ ec2 describe-instances --filters "Name=tag:Project,Values=kiln" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].[Tags[?Key==`Name`]|[0].Value,InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,LaunchTime]' \
    --output text
}

cmd_run() {
  local name="$1" script="$2" id; id=$(instance_id "$name")
  [ "$id" = "None" ] && { echo "no instance named $name" >&2; exit 1; }
  local params; params=$(python3 -c 'import json,sys; print(json.dumps({"commands":[sys.argv[1]],"executionTimeout":[sys.argv[2]]}))' "$script" "${KILN_RUN_TIMEOUT:-3600}")
  local cid; cid=$(aws_ ssm send-command --instance-ids "$id" --document-name AWS-RunShellScript \
    --parameters "$params" --timeout-seconds 600 --query Command.CommandId --output text)
  while :; do
    local st; st=$(aws_ ssm get-command-invocation --command-id "$cid" --instance-id "$id" --query Status --output text 2>/dev/null || echo Pending)
    case "$st" in Pending|InProgress|Delayed) sleep 3;; *) break;; esac
  done
  aws_ ssm get-command-invocation --command-id "$cid" --instance-id "$id" \
    --query '[Status,StandardOutputContent,StandardErrorContent]' --output text
}

cmd_sync() {
  local name="$1" root; root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
  # Replacing $SRC under a running job swaps code beneath it (and spawned ranks re-import
  # it). Refuse unless KILN_FORCE_SYNC=1. Jobs run as `python $SRC/<file>` (cmd_py).
  if [ "${KILN_FORCE_SYNC:-0}" != 1 ]; then
    local busy; busy=$(cmd_run "$name" "pgrep -af 'python ($SRC/)?(bench|tools)/' | grep -v pgrep | head -3" | tail -n +1)
    if echo "$busy" | grep -qE "python ($SRC/)?(bench|tools)/"; then
      echo "refusing to sync $name: a Kiln job is running (KILN_FORCE_SYNC=1 to override):" >&2
      echo "$busy" >&2
      exit 3
    fi
  fi
  local tarball; tarball=$(mktemp -t kiln-src).tgz
  # Tracked AND untracked-but-not-ignored files, so work in progress ships too.
  (cd "$root" && git ls-files -co --exclude-standard -z | xargs -0 tar czf "$tarball")
  local key; key="$name$(echo "$SRC" | tr / -)"
  s3_ cp --quiet "$tarball" "s3://$BUCKET/src/$key.tgz"
  rm -f "$tarball"
  cmd_run "$name" "set -e; mkdir -p $SRC && cd $SRC && aws s3 cp --quiet s3://$BUCKET/src/$key.tgz /tmp/$key.tgz --region $BUCKET_REGION && find $SRC -mindepth 1 -maxdepth 1 ! -name .cache -exec rm -rf {} + && tar xzf /tmp/$key.tgz && echo synced \$(git -C $SRC rev-parse --short HEAD 2>/dev/null || true) \$(find . -type f | wc -l) files" >/dev/null
}

py_env() {
  local venv="${KILN_BOX_VENV:-$VENV}"
  echo "export PATH=$venv/bin:/opt/aws/neuron/bin:\$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=$SRC ${KILN_PY_ENV:-}"
}

py_args() { local args=""; for a in "$@"; do args+=" $(printf %q "$a")"; done; echo "$args"; }

cmd_py() {
  local name="$1" file="$2"; shift 2
  cmd_sync "$name"
  local args; args=$(py_args "$@")
  # The job runs in /opt/kiln/work, not in the source tree: neuronx-cc creates its scratch
  # directories in the cwd, and a sync that replaces the tree would delete them mid-compile
  # (measured: "No such file or directory: '/opt/kiln/src/neuronxcc-...'", exit 70).
  # SSM keeps only the first 24 KB of stdout, so the full log stays on the box (and in S3)
  # and only its tail comes back, one line per row and capped in width.
  local log="/opt/kiln/logs/$(date -u +%Y%m%dT%H%M%SZ)-$(basename "$file" .py).log"
  cmd_run "$name" "mkdir -p /opt/kiln/logs; $(py_env); mkdir -p /opt/kiln/work && cd /opt/kiln/work && python $SRC/$file$args > $log 2>&1; rc=\$?; aws s3 cp --quiet $log s3://$BUCKET/logs/$name/ --region $BUCKET_REGION; echo log=$log rc=\$rc; grep -vE '${KILN_LOG_DROP:-Warning|warn|INFO}' $log | tail -n \${KILN_TAIL:-60} | cut -c1-${KILN_COLS:-400}"
}

cmd_bg() {
  # Like py, but detached: SSM returns at once and the job outlives any client timeout.
  # The exit code lands in <log>.rc and the log is copied to S3 when the job ends.
  local name="$1" file="$2"; shift 2
  cmd_sync "$name"
  local args; args=$(py_args "$@")
  local log="/opt/kiln/logs/$(date -u +%Y%m%dT%H%M%SZ)-$(basename "$file" .py).log"
  local job="$(py_env); cd /opt/kiln/work && python $SRC/$file$args > $log 2>&1; echo \$? > $log.rc; aws s3 cp --quiet $log s3://$BUCKET/logs/$name/ --region $BUCKET_REGION"
  cmd_run "$name" "mkdir -p /opt/kiln/logs /opt/kiln/work; setsid nohup bash -c $(printf %q "$job") > /dev/null 2>&1 < /dev/null & echo started log=$log"
}

cmd_log() {
  local name="$1" log="$2" n="${3:-${KILN_TAIL:-40}}"
  cmd_run "$name" "if [ -f $log.rc ]; then echo rc=\$(cat $log.rc); else echo running; fi; grep -vE '${KILN_LOG_DROP:-Warning|warn|INFO}' $log | tail -n $n | cut -c1-${KILN_COLS:-400}"
}

cmd_down() {
  local ids
  if [ "$1" = "--all" ]; then
    ids=$(aws_ ec2 describe-instances --filters "Name=tag:Project,Values=kiln" \
      "Name=instance-state-name,Values=pending,running,stopping,stopped" \
      --query 'Reservations[].Instances[].InstanceId' --output text)
  else
    ids=$(instance_id "$1"); [ "$ids" = "None" ] && ids=""
  fi
  [ -z "$ids" ] && { echo "nothing to terminate"; return; }
  aws_ ec2 terminate-instances --instance-ids $ids --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text
}

case "${1:-}" in
  bootstrap) cmd_bootstrap ;;
  up) shift; cmd_up "$@" ;;
  ls) shift; cmd_ls "$@" ;;
  ready) shift; cmd_ready "$@" ;;
  run) shift; cmd_run "$@" ;;
  sync) shift; cmd_sync "$@" ;;
  py) shift; cmd_py "$@" ;;
  bg) shift; cmd_bg "$@" ;;
  nvme) shift; cmd_nvme "$@" ;;
  log) shift; cmd_log "$@" ;;
  down) shift; cmd_down "$@" ;;
  *) sed -n "2,/^set -euo/p" "$0" | grep "^#"; exit 2 ;;
esac
