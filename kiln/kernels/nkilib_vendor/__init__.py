"""A vendored subset of AWS's nki-library (nkilib), Apache-2.0: the segmented prefill attention kernel vllm-neuron 0.24
runs on trn2 (vllm_neuron/functional/attention/attention_segmented_cte.py), with the modules it imports.

Origin: nkilib 0.0.0.0dev0+3b542be2 (build Jul 15 2026 16:15:10 UTC), as installed in the AWS Neuron DLAMI SDK 2.32
venv /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0 (ami-01f66e576e60931ed, ap-southeast-4), the copy
vllm-neuron 0.24.0.1.1.0 imports; the public repository is https://github.com/aws-neuron/nki-library (its main at
92d11f6 differs in these files). LICENSE and NOTICE beside this file are that repository's.

Files are byte-identical copies except core/attention/attention_segmented_cte.py and core/attention/attention_cte.py,
marked "Modified by Kiln" (a KV layout flag and NeuronCore-v2 fallbacks, see their headers; every changed line carries
"kiln:"). Kiln calls them through kernels/segmented_attn.py.
"""
