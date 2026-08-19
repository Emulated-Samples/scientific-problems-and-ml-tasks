"""Offline HF-hub fallback for grade time — imported by the root conftest.

The task image bakes the pinned ``gelu-2l`` weights and tokenizer into the
HF hub cache at build time (see environment/warmup-hf-cache.py), and the
verifier grades with network disabled and ``HF_HUB_OFFLINE=1``. Weights,
config, and tokenizer all resolve from that cache natively — but
``transformer_lens.loading_from_pretrained`` resolves NeelNanda/* weight
filenames through a live ``HfApi.list_repo_files()`` call that has no cache
fallback, so it dies offline even with a fully populated cache. When offline
mode is enabled, answer that call from the local cache snapshot instead.

Active only under ``HF_HUB_OFFLINE`` (grade time); online runs untouched.
"""

from pathlib import Path


def _install() -> None:
    try:
        import huggingface_hub
        from huggingface_hub import constants
    except ImportError:  # HF not installed — nothing to patch
        return
    if not constants.HF_HUB_OFFLINE:
        return

    def _snapshot_files(repo_id: str, revision) -> list | None:
        repo_dir = Path(constants.HF_HUB_CACHE) / (
            "models--" + repo_id.replace("/", "--")
        )
        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir():
            return None
        commit = None
        if revision and (snapshots / str(revision)).is_dir():
            commit = str(revision)
        else:
            ref = repo_dir / "refs" / (str(revision) if revision else "main")
            if ref.is_file():
                commit = ref.read_text().strip()
        if not commit or not (snapshots / commit).is_dir():
            return None
        root = snapshots / commit
        return sorted(
            str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
        )

    original = huggingface_hub.HfApi.list_repo_files

    def list_repo_files_offline(self, repo_id, *, revision=None, **kwargs):
        files = _snapshot_files(repo_id, revision)
        if files is not None:
            return files
        return original(self, repo_id, revision=revision, **kwargs)

    huggingface_hub.HfApi.list_repo_files = list_repo_files_offline


_install()
