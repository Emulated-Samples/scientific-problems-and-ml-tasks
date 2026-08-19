# Batch-record HA OpLog — agreed implementation semantics

These notes record the reviewed implementation decisions for the batch-record
HA OpLog design (see `rfc-batch-oplog.md`), including the places where the
implementation intentionally differs from the original RFC text. Where these
notes and the RFC disagree, these notes win. The codec byte format is
specified separately in `wire-format.md`.

The implementation preserves the existing logical `OpLogEntry` and standby
promotion model while replacing the underlying per-entry durable write
protocol with batch records, an ordered writer, and a transactionally
advanced durable prefix. It integrates the protocol with current
MasterService mutation paths, tenant-aware metadata, segment lifecycle
replay, snapshot bootstrap, failure recovery, and promotion catch-up.

## Batch-record protocol

- `HaKvBackend` is the generic KV adapter; an etcd txn/CAS implementation
  backs it in production.
- Multiple continuous `OpLogEntry` values are stored in one
  `OpLogBatchRecord`.
- `{batch_id, last_seq}` advances in `durable_prefix` atomically with the
  batch record.
- Schema versions, checksums, sequence ranges, tenant IDs, cluster IDs, and
  canonical batch keys are validated at storage and codec boundaries.
- txn/CAS support is required. Non-transactional backends are not supported
  by batch-record HA.

## Ordered writer

- `OrderedOpLogWriter` exposes `Reserve()`, `Commit()`, and `Abort()`.
- Sequence IDs are assigned during `Commit()` so aborted reservations do not
  create gaps.
- The writer maintains one backend write in flight and one open waiting
  batch; only the open batch counts against `max_entries_per_batch`.
- Batch seal is work-conserving: a `Commit()` seals immediately whenever no
  write is in flight (even before `Start()`), so commits against an idle
  writer produce single-entry batches, and a commit frees its open-batch
  slot as soon as it seals.
- New admission stops while the current batch cannot be persisted; the
  blocked batch is retried without allowing later entries to pass it. When
  the blocked batch finally persists, admission reopens. Closing admission
  gates `Reserve()` only: a caller already holding a reservation may still
  `Commit()` it while the writer is not accepting (only `Stop()` is
  terminal for existing reservations).
- Durable callbacks are dispatched in logical sequence order after the
  durable prefix advances.
- Entry validation happens in `Commit()`: an entry with an out-of-range
  `op_type`, an invalid tenant ID, or an oversized key/payload is rejected
  with `INVALID_PARAMS`, consumes no sequence, and never fires its
  callback.
- `Reservation` is RAII: destroying (or move-assigning over) an unused
  reservation releases its slot exactly like `Abort()`.
- `Stop()` waits for the in-flight write and drains its callbacks, then
  closes admission permanently: after `Stop()`, `IsAccepting()` is false
  and both `Reserve()` and `Commit()` — including commits of reservations
  taken before the stop — fail with `UNAVAILABLE_IN_CURRENT_STATUS`. (This
  is stricter than the RFC's discussion of draining already-admitted
  work.)
- Error codes: `Reserve()` on a full open batch →
  `TASK_PENDING_LIMIT_EXCEEDED`; `Reserve()`/`Commit()` while not accepting
  (backend-unhealthy, stopped, or never started) →
  `UNAVAILABLE_IN_CURRENT_STATUS`. `LastError()` reports the most recent
  failure cause: the propagated `WriteBatchFn` error after a backend
  failure, or `INVALID_PARAMS` when the configured initial durable prefix
  cannot advance (`last_seq == UINT64_MAX` is rejected up front and the
  writer starts unhealthy).

## Batch storage

- Key layout and range construction are in `wire-format.md`.
- The batch write is a single transaction: exactly one compare —
  `kValueEquals` on the durable-prefix key against the encoded expected
  prefix, or `kKeyNotExists` when creating the first prefix — plus puts of
  the batch-record key and the new encoded prefix.
- `InitDurablePrefix` on a fresh namespace creates `{batch_id=0,
  last_seq=0}` with a create-if-absent transaction; losing that race
  re-reads and returns the winner's prefix.
- **Legacy namespaces are rejected, not migrated.** The batch-record
  implementation is batch-only: `InitDurablePrefix` fails with
  `INCOMPLETE_OPLOG_CATCH_UP` if the namespace contains a legacy `latest`
  key, any fixed-width per-entry key directly under `/oplog/{cluster}/`,
  or any key under `/oplog/{cluster}/snapshot/`. Operators reset legacy
  namespaces by hand (see the deployment guide).
- Unknown-outcome recovery applies only to a failed transaction
  (`ETCD_TRANSACTION_FAIL`): re-read the prefix; if it already equals the
  new prefix and the stored batch record decodes to exactly the batch that
  was written, the write is treated as successful. Other backend errors
  propagate unchanged.
- Error codes: invalid cluster ID or a backend without transaction support
  → `INVALID_PARAMS`; locally-detected invalid input (malformed batch
  shape, discontinuity against `expected_prefix`, prefix fields at
  `UINT64_MAX`) → `INVALID_PARAMS`, checked before any backend call;
  decode/checksum failures on read → `INTERNAL_ERROR`; a durable prefix
  that is missing while batch records exist → `INTERNAL_ERROR`; backend
  errors (`ETCD_OPERATION_ERROR`, `ETCD_KEY_NOT_EXIST`,
  `ETCD_TRANSACTION_FAIL`) propagate unchanged.
- `ReadBatchesAfter` ignores keys whose final path segment is not exactly
  20 digits; `limit` counts decoded batch records. It pages through the
  backend by re-issuing `Range` with `begin = <last returned key> + '\0'`
  and `limit = <remaining>` until the limit is met or a short page comes
  back.

## Standby reader and applier

- The standby polls the durable prefix; there is no watch and no legacy
  per-entry fallback.
- `PollOnce` dispositions: backend transport errors on `Get`/`Range` →
  `RETRYABLE` with the propagated code; every consistency violation →
  `FATAL`. A missing durable prefix before one has ever been observed is
  `OK` with `durable_prefix_present = false`; a prefix that disappears or
  regresses after being observed is `FATAL`.
- The prefix must land on a batch boundary: if the batch named by
  `prefix.batch_id` has `last_seq != prefix.last_seq`, the poll fails
  before applying anything. A prefix of `{batch_id = 0, last_seq > 0}` is
  invalid (batch 0 implies sequence 0), and a first available batch whose
  `first_seq` does not match the applier's expected next sequence is a
  boundary violation — both are `FATAL` with `INCOMPLETE_OPLOG_CATCH_UP`.
  A prefix that references batches which do not exist in the range is
  `FATAL` with `ETCD_KEY_NOT_EXIST`.
- If the prefix has not advanced past what is already applied, `PollOnce`
  returns without issuing any range read.
- Batches are read sequentially from the earliest available batch. Entries
  below the applier's expected sequence are skipped silently;
  `applied_entries` counts only newly applied entries. A gap or overlap in
  a later batch fails the poll, but batches already applied stay applied.
  `max_batches` caps the batches consumed per poll; the remainder is
  picked up by the next poll.
- The applier is strictly sequential with no buffering: an entry beyond
  the expected next sequence returns false and is discarded (no pending
  queue, no gap resolution, no timeout-based skipping); an entry below it
  is a no-op returning true. Apply is tolerant of payload content: a
  PUT_END with an empty or undecodable payload still applies with
  default metadata, and a garbage segment payload advances the sequence
  without mutating the registry. The segment registry is keyed by
  `transport_endpoint`.

## MasterService integration

The batch writer is integrated with `PutEnd`, replica updates, segment
lifecycle operations, offload/promotion completion, remove variants, replica
clear, eviction, and expired cleanup paths.

Visibility and resource semantics:

- Additive operations (`PutEnd`, replica additions, `CopyEnd`,
  `MoveEnd`-target, offload fallback, promotion success, segment mount)
  become visible on the primary and return before durability.
- Destructive operations (remove variants, replica clears, eviction,
  `MoveEnd`-source, stale/expired cleanup) first mark affected replicas as
  `REMOVED` and also **return success immediately** without waiting for
  durability.
- `REMOVED` replicas are excluded from reads immediately; memory, quota,
  local-disk accounting, and metadata resources are released only from the
  ordered durable callback.
- OpLog encoding of mutations: a full object removal logs `REMOVE`; a
  mutation that leaves readable replicas behind logs `PUT_END` carrying the
  full post-state metadata payload (never empty); segment lifecycle logs
  `SEGMENT_MOUNT` / `SEGMENT_UNMOUNT`.
- Direct legacy OpLog writes are rejected while batch mode is active.
- Fail-closed behavior: after the first backend transaction failure,
  appends fail with `UNAVAILABLE_IN_CURRENT_STATUS` (reads keep working).
  When no reservation slot is available, destructive operations fail
  **before any mutation** with `TASK_PENDING_LIMIT_EXCEEDED` and the
  object stays readable.
- `ExistKey` means "has a COMPLETE replica" and reports false rather than
  an error. Batch-removing an object whose replicas are all `REMOVED`
  reports `OBJECT_NOT_FOUND` for that key.
- If the production OpLog writer cannot be initialized at construction
  (enable_oplog with an etcd backend that fails), the `MasterService`
  constructor throws `std::runtime_error`. With `enable_oplog` on a
  non-etcd HA backend, no writer is created (the flag is inert).

## Restore and remount

`RestoreFromStandbySnapshot` rebuilds a promoted primary from a
`PromotionContext`:

- Replica descriptors are preserved exactly as exported.
- Restored **memory** replicas are not readable until their segment is
  remounted: reads on them return `REPLICA_IS_NOT_READY`. Their bytes are
  pre-accounted in the allocated-memory metric while awaiting remount.
- Restored NoF/local-disk replicas are readable immediately and are exempt
  from endpoint filtering.
- Replica endpoints whose segments are not in the restored registry are
  filtered out of `GetReplicaList` results.
- `cxl` segment endpoints are rewritten to the segment name at restore and
  can never be remounted; attempting it fails with
  `UNAVAILABLE_IN_CURRENT_MODE`.
- Standby remount (`ReMountSegment` on restored state) validates the
  incoming segment against the restored registry — name, endpoint, and
  base/size must match, addresses must not duplicate or overlap another
  segment, and the caller must own the segment — failing with
  `INVALID_PARAMS` (this supersedes the legacy "unsolvable remount errors
  return OK" note). Multi-segment remount is all-or-nothing: one invalid
  segment publishes neither, and the call is retryable after the conflict
  is fixed.

Allocator-state restore (`RestoreOffsetBufferAllocator` /
`RestoreCachelibBufferAllocator`):

- Live allocations are reconstructed at their **exact original
  addresses**; the output handle list preserves **input order**.
- Address gaps stay occupied (counted in the allocator's used size) so new
  allocations can never overlap a restored range; releasing a restored
  handle frees exactly its address for reuse. There is no cap on the
  number of gaps.
- Rejected inputs: endpoint mismatch, duplicate or overlapping
  descriptors, addresses out of range (address + size beyond base +
  capacity, or arithmetic overflow), size normalization overflowing
  capacity; for cachelib additionally misaligned bases and chunks starting
  in a slab's unusable tail. A failed restore leaks no state — a valid
  restore still works afterwards.
- Cachelib restore is memory-only: `NOF_SSD` replicas and `cxl` protocol
  descriptors are rejected; `rdma` is accepted.

## Standby bootstrap and promotion

- The existing snapshot baseline is restored before OpLog catch-up; catch-up
  starts from the snapshot's `last_included_seq` (or 1 with no snapshot).
- The standby catches up to the latest durable prefix before promotion and
  fails closed if the prefix is unreachable or inconsistent.
- Transient standby backend errors are retried with exponential backoff
  and reset `StandbySyncStatus::last_error` to `OK` on recovery. After
  `batch_oplog_retry_timeout_sec` (180 seconds by default) of consecutive
  retryable failures the standby enters `FAILED` with the last backend
  error code in `last_error`; it can be restarted via `Start()`.
- The promotion catch-up target is the durable prefix **re-read at
  promotion time** (even if a prefix first appeared after the standby
  reached WATCHING). One transient prefix read failure is retried within
  the promotion deadline. Catch-up must paginate when more than 1024
  batches are outstanding.
- Promotion fails with `INCOMPLETE_OPLOG_CATCH_UP` when the durable
  prefix or a referenced batch is unreadable, or when the prefix is
  missing while the locally applied sequence is nonzero. A failed
  promotion leaves the standby `FAILED`; a successful one leaves it
  `STOPPED`.
- `StandbyController::PromoteStandbyAndExport` before `Start()` fails with
  `UNAVAILABLE_IN_CURRENT_STATUS` when an OpLog reader exists
  (`enable_oplog` with the etcd backend); with no reader (oplog off, or a
  non-etcd backend) it succeeds. After a failed `Start()`, the start error
  is propagated. In builds without etcd support, starting the batch
  standby fails with `INTERNAL_ERROR` and the FAILED state.
- `SetCatchUpBatchKvBackendForTesting` injects the backend for **both**
  the polling loop and promotion catch-up, allowing `Start()` to run
  without a real etcd.
- Promotion exports a `PromotionContext` (applied sequence, object
  metadata, segment registry); the new primary restores metadata shards,
  the segment registry, and allocator state from it (see Restore and
  remount).

## Metrics

`HAMetricManager` serializes each metric under `ha_` + the member name
(e.g. `ha_oplog_last_sequence_id`, `ha_batch_record_retry_total` —
singular `retry_total`). The batch-record metrics are compiled in only
under `MOONCAKE_ENABLE_OPLOG_PERF_METRICS` and are absent from the
serialized output when the flag is off. The human-readable summary string
includes the `last_seq` and `applied_seq` figures.

## Test failpoints

`TestFailPoint::Wait(name)` implements a file-handshake failpoint for
integration tests, compiled in only under `MOONCAKE_ENABLE_TEST_FAILPOINTS`
(otherwise it always returns false immediately). Failpoint names use
`[a-z0-9_]` only. With the feature compiled in:

- The directory comes from `MOONCAKE_TEST_FAILPOINT_DIR` (unset → return
  false immediately); the poll timeout in seconds from
  `MOONCAKE_TEST_FAILPOINT_TIMEOUT_SEC` (default 30, clamped to 1–3600).
- `Wait` claims an armed failpoint by atomically renaming `<name>.arm` to
  `<name>.claimed.<pid>`; if the rename fails (not armed), it returns
  false immediately.
- After claiming, it removes any stale `<name>.release`, then publishes
  `<name>.hit` (write to a temporary name, then rename), and polls (~10 ms
  interval) for `<name>.release` until the timeout.
- On release it deletes the `.release`, `.hit`, and `.claimed.<pid>` files
  and returns true; on timeout it cleans up `.hit` and `.claimed.<pid>`
  and returns false.

## Configuration and compatibility

- `enable_oplog` turns the batch-record OpLog on; it is off by default and
  requires HA mode with the etcd backend. It must survive every master
  config conversion path.
- Defaults: `oplog_batch_max_entries = 1024` (no clamping),
  `oplog_poll_interval_ms = 1000`, `batch_oplog_retry_timeout_sec = 180`.
- Snapshots taken with the batch OpLog active record
  `last_included_seq = durable_prefix.last_seq`.
- Batch-to-legacy rollback is not supported, and legacy namespaces are not
  readable (see Batch storage above).
- Existing snapshot restore remains supported. Batch primaries currently
  skip snapshot generation; moving snapshot generation to the standby is
  follow-up work.

## Intentional differences from the RFC text

- Legacy per-entry data is rejected rather than cut over: the RFC's
  `/latest`-based cutover initialization was dropped in review in favor of
  a batch-only namespace with an operator-driven reset procedure.
- The writer uses two logical batches (open waiting + sealed/writing)
  rather than a separate written/completed batch.
- Batch lookup is sequential; binary lookup by applied sequence is
  deferred.
- Entries retain the current full-metadata payload instead of introducing a
  delta schema.
- Callback execution remains ordered and asynchronous.
- `Stop()` is terminal for all reservations, including ones taken before
  the stop (the RFC left this open).

## Out of scope (intentionally not part of this change)

- Durable allocation records for `PutStart`, `UpsertStart`, `MoveStart`,
  and `CopyStart`.
- Replayable `put_start_release_timeout_sec` delayed-release state.
- Delta records and replica-level data-block lifecycle management.
- Automatic rejection of legacy startup when batch history exists.
- Leadership fencing and forced failover after extended writer failure.
- Callback-result reporting or retry semantics at the RPC boundary.
- Byte-based batch limits, durable-prefix watch, additional metrics,
  checkpoint lookup, and long-term snapshot/OpLog compaction.
