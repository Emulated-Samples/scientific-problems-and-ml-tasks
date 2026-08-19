"""Build-time HF cache warm-up (offline-verifier fix).

The behavioral-test fixtures load TransformerLens "gelu-2l" at grade time,
and the verifier runs with network disabled (task.toml network_mode
"no-network") plus HF_HUB_OFFLINE=1. Bake the exact model revision into the
image's HF hub cache so grading never needs the network.

snapshot_download(revision=<sha>) does not write refs/main; grade-time code
asks for "main", so point that ref at the pinned commit explicitly.
"""

import os

from huggingface_hub import constants, snapshot_download

PINS = {
    # TransformerLens alias "gelu-2l" -> weights repo (config.json + model_final.pth)
    "NeelNanda/GELU_2L512W_C4_Code": "2ef9ffce191eb048533e4ef1c35e25c5ac4a8c97",
    # tokenizer repo TransformerLens loads for NeelNanda models
    "NeelNanda/gpt-neox-tokenizer-digits": "0f6671571a20be9756b9991d978047c03b75e749",
}

for repo, rev in PINS.items():
    path = snapshot_download(repo, revision=rev)
    repo_dir = os.path.join(
        constants.HF_HUB_CACHE, "models--" + repo.replace("/", "--")
    )
    refs = os.path.join(repo_dir, "refs")
    os.makedirs(refs, exist_ok=True)
    with open(os.path.join(refs, "main"), "w") as f:
        f.write(rev)
    print(f"baked {repo} @ {rev} -> {path}")

print("HF cache warm-up complete")
