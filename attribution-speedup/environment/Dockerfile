FROM amazonlinux:2023

RUN dnf install -y --setopt=install_weak_deps=False --allowerasing \
      curl \
      nodejs20 nodejs20-npm \
      git tar gzip unzip rsync patch procps-ng openssh-clients ca-certificates \
      which \
      python3.11 python3.11-pip python3.11-devel \
      python3.12 python3.12-pip python3.12-devel \
      gcc gcc-c++ make \
    && dnf clean all \
    && rm -rf /var/cache/dnf

RUN if ! command -v node >/dev/null 2>&1; then \
      alternatives --install /usr/bin/node node /usr/bin/node-20 100; \
    fi \
    && if ! command -v npm >/dev/null 2>&1; then \
      alternatives --install /usr/bin/npm npm /usr/bin/npm-20 100; \
    fi \
    && node --version && npm --version

COPY source/env/ /hyperfocal/env/
COPY source/repo.bundle /hyperfocal/git-bundles/repo.bundle
COPY source/git-submodules/ /hyperfocal/git-bundles/git-submodules/
WORKDIR /hyperfocal/env

RUN set -e; \
    git init -q .; \
    git fetch -q /hyperfocal/git-bundles/repo.bundle "+refs/*:refs/*" "+HEAD:refs/hyperfocal/bundle-head" --update-head-ok; \
    git symbolic-ref HEAD refs/heads/harbor-release/problem/base-speedup/base-speedup; \
    git read-tree HEAD; \
    git update-ref -d refs/hyperfocal/bundle-head; \
    git remote add origin "https://github.com/Hyperfocal-AI-Environments/lat-bench"; \
    mkdir -p /hyperfocal/env/.git/modules/packages; \
    git init -q --separate-git-dir /hyperfocal/env/.git/modules/packages/env-base /hyperfocal/env/packages/env-base; \
    git -C /hyperfocal/env/packages/env-base fetch -q /hyperfocal/git-bundles/git-submodules/packages__env-base.bundle "+refs/*:refs/*" "+HEAD:refs/hyperfocal/bundle-head" --update-head-ok; \
    git -C /hyperfocal/env/packages/env-base update-ref --no-deref HEAD 14e43e3b44fe660701c709f28933940ae3cf62bc; \
    git -C /hyperfocal/env/packages/env-base read-tree HEAD; \
    git -C /hyperfocal/env/packages/env-base update-ref -d refs/hyperfocal/bundle-head; \
    git -C /hyperfocal/env config submodule.packages/env-base.url "https://github.com/Hyperfocal-AI-Packages/env-base.git"; \
    git -C /hyperfocal/env config submodule.packages/env-base.active true; \
    mkdir -p /hyperfocal/env/.git/modules/packages; \
    git init -q --separate-git-dir /hyperfocal/env/.git/modules/packages/env-orchestrator /hyperfocal/env/packages/env-orchestrator; \
    git -C /hyperfocal/env/packages/env-orchestrator fetch -q /hyperfocal/git-bundles/git-submodules/packages__env-orchestrator.bundle "+refs/*:refs/*" "+HEAD:refs/hyperfocal/bundle-head" --update-head-ok; \
    git -C /hyperfocal/env/packages/env-orchestrator update-ref --no-deref HEAD 28feff8febd94c65636c145854a3d5fb7b6fbc03; \
    git -C /hyperfocal/env/packages/env-orchestrator read-tree HEAD; \
    git -C /hyperfocal/env/packages/env-orchestrator update-ref -d refs/hyperfocal/bundle-head; \
    git -C /hyperfocal/env config submodule.packages/env-orchestrator.url "https://github.com/Hyperfocal-AI-Packages/env-orchestrator.git"; \
    git -C /hyperfocal/env config submodule.packages/env-orchestrator.active true; \
    rm -rf /hyperfocal/git-bundles; \
    mkdir -p /hyperfocal/logs && chmod 777 /hyperfocal/logs; \
    touch environment/.env; \
    (cd packages/env-base && npm install --no-audit --no-fund && npm run build); \
    (cd packages/env-orchestrator && npm install --no-audit --no-fund && npm run build); \
    if [ -d packages/mock-mcp-services ]; then (cd packages/mock-mcp-services && npm install --no-audit --no-fund && npm run build); fi; \
    (cd environment && npm install --no-audit --no-fund && npm run build); \
    node packages/env-orchestrator/bin/env-orchestrator.js setup --problem base-speedup; \
    useradd --create-home --shell /bin/bash agent; \
    mkdir -p /tests /solution /logs/verifier; \
    chmod 700 /tests; chmod 755 /solution; chmod 777 /logs /logs/verifier; \
    chmod -R o-rwx /hyperfocal/env/.git /hyperfocal/env/environment /hyperfocal/env/packages; \
    chmod o-rwx /hyperfocal/env/hyperfocal.yaml 2>/dev/null || true; \
    chown -R agent:agent /hyperfocal/env/workspace; \
    npm cache clean --force >/dev/null 2>&1 || true

# Offline-verifier fix: bake the pinned gelu-2l weights + tokenizer into the
# image's HF hub cache at build time (needs network — build has egress). At
# grade time the verifier runs no-network with HF_HUB_OFFLINE=1 (task.toml
# [verifier.env]); the fixtures then resolve the model from this cache with
# zero network. Kept as its own layer, after the grade venv is provisioned.
RUN /hyperfocal/env/environment/.grade-venv/bin/python \
      /hyperfocal/env/environment/warmup-hf-cache.py

# Hermeticity gate: prove at BUILD time that the grade-time model load works
# with the hub client in offline mode (same env the verifier grades under).
# If the cache bake or the offline shim ever regresses, the build fails here
# instead of the verifier zeroing tests at grade time.
RUN cd /hyperfocal/env/environment/behavioral-tests && \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /hyperfocal/env/environment/.grade-venv/bin/python -c "\
import sys; sys.path.insert(0, '.'); \
from _lib import hf_offline; \
from transformer_lens import HookedTransformer; \
HookedTransformer.from_pretrained('gelu-2l', device='cpu'); \
print('offline gelu-2l load OK')"

CMD ["sleep", "infinity"]
