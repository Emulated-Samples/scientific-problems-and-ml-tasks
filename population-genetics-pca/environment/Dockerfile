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
    git symbolic-ref HEAD refs/heads/harbor-release/release-from_scratch_pca/from_scratch_pca; \
    git read-tree HEAD; \
    git update-ref -d refs/hyperfocal/bundle-head; \
    git remote add origin "https://github.com/Hyperfocal-AI-Environments/sc-pcabench"; \
    mkdir -p /hyperfocal/env/.git/modules/packages; \
    git init -q --separate-git-dir /hyperfocal/env/.git/modules/packages/env-base /hyperfocal/env/packages/env-base; \
    git -C /hyperfocal/env/packages/env-base fetch -q /hyperfocal/git-bundles/git-submodules/packages__env-base.bundle "+refs/*:refs/*" "+HEAD:refs/hyperfocal/bundle-head" --update-head-ok; \
    git -C /hyperfocal/env/packages/env-base update-ref --no-deref HEAD 17eb75cdf518c108dc77340da75e5cd1ee24f0d6; \
    git -C /hyperfocal/env/packages/env-base read-tree HEAD; \
    git -C /hyperfocal/env/packages/env-base update-ref -d refs/hyperfocal/bundle-head; \
    git -C /hyperfocal/env config submodule.packages/env-base.url "https://github.com/Hyperfocal-AI-Packages/env-base.git"; \
    git -C /hyperfocal/env config submodule.packages/env-base.active true; \
    mkdir -p /hyperfocal/env/.git/modules/packages; \
    git init -q --separate-git-dir /hyperfocal/env/.git/modules/packages/env-orchestrator /hyperfocal/env/packages/env-orchestrator; \
    git -C /hyperfocal/env/packages/env-orchestrator fetch -q /hyperfocal/git-bundles/git-submodules/packages__env-orchestrator.bundle "+refs/*:refs/*" "+HEAD:refs/hyperfocal/bundle-head" --update-head-ok; \
    git -C /hyperfocal/env/packages/env-orchestrator update-ref --no-deref HEAD 60f8ecd36a459914a34182a62ab602c69e887793; \
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
    node packages/env-orchestrator/bin/env-orchestrator.js setup --problem from_scratch_pca; \
    useradd --create-home --shell /bin/bash agent; \
    mkdir -p /tests /solution /logs/verifier; \
    chmod 700 /tests; chmod 755 /solution; chmod 777 /logs /logs/verifier; \
    chmod -R o-rwx /hyperfocal/env/.git /hyperfocal/env/environment /hyperfocal/env/packages \
      /hyperfocal/env/cheats /hyperfocal/env/data /hyperfocal/env/docs /hyperfocal/env/fn_resistance \
      /hyperfocal/env/grader /hyperfocal/env/reference /hyperfocal/env/scripts /hyperfocal/env/tasks \
      /hyperfocal/env/tests; \
    chmod o-rwx /hyperfocal/env/hyperfocal.yaml /hyperfocal/env/README.md /hyperfocal/env/pyproject.toml /hyperfocal/env/uv.lock 2>/dev/null || true; \
    chown -R agent:agent /hyperfocal/env/workspace; \
    npm cache clean --force >/dev/null 2>&1 || true

CMD ["sleep", "infinity"]
