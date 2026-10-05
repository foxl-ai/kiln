"""Every test must leave the process as it found it: the suite runs in ONE pytest process, and state a
test leaves behind reaches every test after it.

Measured (63f683b single-process suite, kiln-cf-4, 2026-10-04): the pytest process had collected 24
gloo listeners from earlier tensor-parallel engines, so engine/tp.py free_port (then probing
127.0.0.1 only) handed tests/test_attention_tp.py test_mla a port one of them held. Rank 0's
TCPStore bind failed, which left torch's group counter at 1 and the spawned worker waiting for its
store. The next tensor-parallel test (test_qwen3_5_gdn) then named its process group "1" against
its workers' "0" and all four ranks waited in gloo's full mesh until the 4-hour timeout. Per-file
runs never showed it.

The guard below fails a test that leaves torch.distributed initialized, a tensor-parallel engine
open, a multiprocessing child alive, torch's group counter off zero, the environment changed or
torch's thread count changed, and puts each of those back so the failure stays with that test.
"""

import multiprocessing
import os

import pytest

# pytest rewrites this one itself around every phase of every test.
_VOLATILE_ENV = {"PYTEST_CURRENT_TEST"}
# Set once per process by a library the first time something imports or uses it, whichever test that
# is: sklearn/__init__.py os.environ.setdefault (transformers imports sklearn lazily) and
# torch/_inductor/runtime/cache_dir_utils.py cache_dir() (the first inductor compile). Exempt only
# when they APPEAR; a test that changes or removes one is still caught.
_LIBRARY_ENV = {"KMP_DUPLICATE_LIB_OK", "KMP_INIT_AT_FORK", "TORCHINDUCTOR_CACHE_DIR"}
# cv2/__init__.py bootstrap (OpenCV, imported by transformers' image processors) PREPENDS its
# library directories to LD_LIBRARY_PATH at its first import ("amending of LD_LIBRARY_PATH works for
# sub-processes only"); exempt exactly that shape, the old value kept as the suffix.
_PREPENDED_ENV = {"LD_LIBRARY_PATH"}


def _library_made(k, old, new) -> bool:
    if k in _LIBRARY_ENV:
        return old is None
    if k in _PREPENDED_ENV and new is not None:
        return new.endswith(old or "")
    return False


@pytest.fixture(autouse=True)
def _leaves_no_process_state(request):
    import torch
    import torch.distributed as dist

    env = dict(os.environ)
    children = set(multiprocessing.active_children())
    threads = torch.get_num_threads()
    yield
    problems = []
    try:
        from kiln.engine import engine as _engine

        left = list(getattr(_engine, "_OPEN", ()))
    except ImportError:
        left = []
    for eng in left:
        problems.append(f"left a tensor-parallel LLMEngine open (tp={eng.cfg.tp}); close() it")
        eng.close()
    if dist.is_available() and dist.is_initialized():
        problems.append("left torch.distributed initialized")
        dist.destroy_process_group()
    if dist.is_available():
        from torch.distributed import distributed_c10d as c10d

        if c10d._world.group_count:
            problems.append(f"left torch's process-group counter at {c10d._world.group_count} with no group "
                            "(a failed init_process_group: the next init in this process would hang)")
            c10d._world.group_count = 0
    for p in set(multiprocessing.active_children()) - children:
        problems.append(f"left a child process alive ({p.name}, pid {p.pid})")
        p.terminate()
        p.join(timeout=10)
    now = dict(os.environ)
    changed = sorted(k for k in set(env) | set(now) if k not in _VOLATILE_ENV and env.get(k) != now.get(k)
                     and not _library_made(k, env.get(k), now.get(k)))
    if changed:
        problems.append("changed the environment: " + ", ".join(f"{k} {env.get(k)!r:.60} -> {now.get(k)!r:.60}"
                                                                for k in changed) + " (use monkeypatch.setenv)")
        for k in changed:
            if k in env:
                os.environ[k] = env[k]
            else:
                os.environ.pop(k, None)
    if torch.get_num_threads() != threads:
        problems.append(f"changed torch's thread count {threads} -> {torch.get_num_threads()}")
        torch.set_num_threads(threads)
    if problems:
        pytest.fail(f"{request.node.nodeid}: " + "; ".join(problems), pytrace=False)
