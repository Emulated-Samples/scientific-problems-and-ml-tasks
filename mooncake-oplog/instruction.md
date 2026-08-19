I need the batch-record HA OpLog subsystem implemented. `/workspace/Mooncake` is Mooncake Store, our distributed KV-cache storage engine. I'm moving the master's HA hot-standby replication from per-entry OpLog keys to the batch-record protocol specified in:

- `/workspace/specs/rfc-batch-oplog.md`: the design RFC
- `/workspace/specs/implementation-notes.md`: the reviewed implementation semantics, including intentional deviations from the RFC
- `/workspace/specs/wire-format.md`: the codec byte format
- the High Availability section of `docs/source/deployment/mooncake-store-deployment-guide.md`

The interface contract has already landed: the headers under `mooncake-store/include/`, the build wiring, the config surface, and the docs are final. The implementation has not: the new sources are empty stubs, the legacy per-entry implementation is removed, and the sources that integrate with the new interfaces are untouched, so the tree does not build until the work is done.

Implement the subsystem so the store builds with the default CMake configuration and behaves exactly as the specs say, from the codec up through the master and hot-standby integration. Treat the headers as a fixed contract, do not modify anything under `mooncake-store/tests/`, and do not regress existing behavior. A prebuilt build tree at `/workspace/Mooncake/build` keeps rebuilds incremental.

You have 24 hours.
