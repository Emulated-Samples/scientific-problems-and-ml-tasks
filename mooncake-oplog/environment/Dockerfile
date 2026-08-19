# Mooncake Store staged for the batch-record HA OpLog task: parent commit
# warm-built first (so the agent's edit-compile-test loop is incremental),
# then the merged interface contract is overlaid with newer mtimes and the
# implementation is withheld behind stubs.
ARG BASE=ubuntu:22.04@sha256:3b06811b2afd352be909dd088a004166d665dc76d38b13eada33522a9d915c6f

FROM ${BASE}

# System packages: the ubuntu list from upstream dependencies.sh (minus Go and
# SPDK, which this build does not use), plus agent tooling.
ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git wget unzip ca-certificates \
        libibverbs-dev libgoogle-glog-dev libjsoncpp-dev libunwind-dev \
        libnuma-dev libpython3-dev libboost-all-dev libssl-dev \
        libgrpc-dev libgrpc++-dev libprotobuf-dev libyaml-cpp-dev \
        protobuf-compiler-grpc libcurl4-openssl-dev libhiredis-dev \
        liburing-dev libjemalloc-dev libmsgpack-dev libzmq3-dev \
        libzstd-dev libasio-dev libxxhash-dev pkg-config patchelf \
        libc6-dev libc-bin libaio-dev \
        python3 procps gdb vim less && \
    rm -rf /var/lib/apt/lists/*

# Go toolchain for the etcd wrapper (go.mod pins go 1.25 / toolchain
# go1.25.10; GOTOOLCHAIN=local forbids surprise toolchain downloads), plus
# the etcd server binary upstream CI runs its etcd-backed tests against.
RUN ARCH=$(dpkg --print-architecture) && \
    wget -q https://go.dev/dl/go1.25.10.linux-${ARCH}.tar.gz -O /tmp/go.tgz && \
    tar -C /usr/local -xzf /tmp/go.tgz && rm /tmp/go.tgz && \
    wget -q https://github.com/etcd-io/etcd/releases/download/v3.6.1/etcd-v3.6.1-linux-${ARCH}.tar.gz -O /tmp/etcd.tgz && \
    tar -C /tmp -xzf /tmp/etcd.tgz && \
    mv /tmp/etcd-v3.6.1-linux-${ARCH}/etcd /tmp/etcd-v3.6.1-linux-${ARCH}/etcdctl /usr/local/bin/ && \
    rm -rf /tmp/etcd.tgz /tmp/etcd-v3.6.1-linux-${ARCH}
ENV PATH=/usr/local/go/bin:${PATH}
ENV GOTOOLCHAIN=local

# Source tree at the task's starting commit. History and upstream pointers are
# removed: the workspace is a plain source tree, not a git checkout.
ARG PARENT_COMMIT=d06806d19aba7d67712245fa399f4c33191060c4
RUN git clone https://github.com/kvcache-ai/Mooncake.git /workspace/Mooncake && \
    cd /workspace/Mooncake && \
    git checkout --quiet ${PARENT_COMMIT} && \
    git submodule update --init --depth 1 extern/pybind11 extern/yalantinglibs && \
    rm -rf .git extern/pybind11/.git extern/yalantinglibs/.git \
        .github .claude .claude-plugin AGENTS.md CLAUDE.md mooncake-rl

# Mooncake's cmake locates yalantinglibs via find_package: build and install
# it from the submodule exactly as upstream dependencies.sh does.
RUN cd /workspace/Mooncake/extern/yalantinglibs && \
    mkdir -p build && cd build && \
    cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF && \
    cmake --build . -j"$(nproc)" && \
    cmake --install . && \
    rm -rf /workspace/Mooncake/extern/yalantinglibs/build

# Warm build of the PARENT tree (CPU-only, STORE_USE_ETCD=ON exactly like
# upstream store CI — the batch standby hard-requires the flag; the Go etcd
# wrapper builds here with network for `go mod tidy`). The dependency
# closure of these targets covers everything the graded binaries need:
# mooncake_store, transfer_engine, cachelib, the etcd wrapper, gtest
# (FetchContent, cached in build/_deps), and both test-macro link paths.
RUN cd /workspace/Mooncake && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_STORE=ON -DWITH_TE=ON \
        -DWITH_STORE_RUST=OFF -DWITH_P2P_STORE=OFF -DBUILD_UNIT_TESTS=ON \
        -DBUILD_EXAMPLES=OFF -DSTORE_USE_ETCD=ON && \
    make -j"$(nproc)" master_service_test buffer_allocator_test \
        standby_state_machine_test && \
    ./mooncake-store/tests/buffer_allocator_test --gtest_brief=1 && \
    ./mooncake-store/tests/master_service_test --gtest_list_tests > /dev/null

# Flatten mtimes of the warm-built parent state.
RUN find /workspace -exec touch -h -d '2026-08-01T00:00:00Z' {} +

# Overlay the merged interface contract: headers, build wiring, config
# surface, and the deployment-guide HA documentation land verbatim; the
# legacy oplog implementation and every withheld test file are removed; the
# new implementation files are empty stubs. Everything the overlay touches
# gets a NEWER mtime than the warm objects so the first rebuild picks up
# every dependent translation unit.
COPY contract.patch removed-files.txt stub-sources.txt tests-CMakeLists-agent.txt /tmp/stage/
COPY specs/ /workspace/specs/
RUN touch /tmp/stage/stamp && \
    cd /workspace/Mooncake && \
    patch -p1 -s < /tmp/stage/contract.patch && \
    xargs rm -f < /tmp/stage/removed-files.txt && \
    while read -r f; do \
        mkdir -p "$(dirname "$f")"; \
        case "$f" in \
          */metadata_store.cpp)     inc="metadata_store.h" ;; \
          */etcd_ha_kv_backend.cpp) inc="ha/kv/etcd_ha_kv_backend.h" ;; \
          *) inc="ha/oplog/$(basename "${f%.cpp}").h" ;; \
        esac; \
        printf '#include "%s"\n' "$inc" > "$f"; \
    done < /tmp/stage/stub-sources.txt && \
    cp /tmp/stage/tests-CMakeLists-agent.txt mooncake-store/tests/CMakeLists.txt && \
    find /workspace -newer /tmp/stage/stamp \
        -exec touch -h -d '2026-08-02T00:00:00Z' {} + && \
    find /workspace/specs -exec touch -h -d '2026-08-02T00:00:00Z' {} + && \
    rm -rf /tmp/stage

# The overlay updated etcd_wrapper.go (Txn support): rebuild the Go wrapper
# now, while network is available, and stamp its outputs newer than the
# source so no offline rebuild is ever triggered at verify time.
RUN cd /workspace/Mooncake/build && make build_etcd_wrapper && \
    find /workspace/Mooncake/build/mooncake-common/etcd \
        /workspace/Mooncake/mooncake-common/etcd \
        -exec touch -h -d '2026-08-03T00:00:00Z' {} +

RUN useradd --create-home --shell /bin/bash agent && \
    chown -R agent:agent /workspace && \
    install -d -m 755 -o agent -g agent /workspace/scratch && \
    install -d -m 755 /logs && \
    install -d -m 755 -o agent -g agent /logs/artifacts && \
    install -d -m 700 /tests

WORKDIR /workspace
