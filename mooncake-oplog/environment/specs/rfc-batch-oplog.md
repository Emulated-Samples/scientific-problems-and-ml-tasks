# RFC: OpLog Batch Record Ordered Writer and Durable Prefix

## Introduction

Mooncake Store HA should do more than restart the Master process quickly. After a primary failure and failover, the new Master should preserve as much useful KV cache metadata, segment registry state, and object-location index state as possible.

The existing HA work has two complementary recovery paths:

1. **Snapshot** provides a coarse-grained baseline for restoring complete Master state.
2. **OpLog** provides incremental changes after the snapshot, so a standby can keep following the primary and catch up before promotion.

RFC #1200 proposed an etcd-backed hot-standby mechanism for Master metadata HA. The primary writes metadata mutations as OpLog entries, and the standby reads/watches those entries from etcd and applies them locally. RFC #1650 then introduced an `OpLogStore` / `OpLogChangeNotifier` abstraction to decouple the upper-layer replication logic from the etcd-specific implementation.


This RFC builds on #2331 and proposes the next step for the OpLog write and replication protocol: **batch records**, an **ordered writer**, and a **durable prefix**. The goal is not to replace the business semantics that #2331 already adapted. Instead, this RFC keeps the current logical `OpLogEntry` semantics and replaces the underlying write and replay protocol to address lock-held durable waits, per-entry backend key amplification, and promotion boundary ambiguity.

## Background

### Roadmap Context

The Mooncake Store V3 Roadmap (#1035) tracks HA under `Milestone 2: Master Service Enhancements`, specifically `Metadata Adaptation - HA`. That item has three related lines of work:

- **Snapshot**: #1150, #1431, #1465, #1381 and related work provide persistence and recovery for Master metadata, segment state, allocator state, and task-manager state.
- **OpLog**: #1200, #1451, #1515 and related work provide incremental metadata replication after a snapshot baseline.
- **HA refactoring**: #1648, #1678, #1722, #1777 and related work move leader coordination, standby runtime, snapshot bootstrap, and OpLog following into clearer runtime boundaries.

The relationship is straightforward: snapshot restores a complete baseline, OpLog catches the standby up from that baseline to a later primary state, and the HA runtime orchestrates snapshot bootstrap, OpLog following, promotion, and serving.

### Snapshot Baseline

RFC #1150 was motivated by a practical limitation of early HA mode: fast Master restart alone does not recover KV cache location metadata. If all metadata is lost after failover, upper-layer inference must rebuild cache state, and the recovery time can be much longer than the Master process restart time.

PR #1431 implemented Master snapshot and restore:

- fork + copy-on-write to reduce blocking during snapshot creation;
- background serialization and compression of metadata, segment state, allocator state, and task-manager state;
- local filesystem and S3 snapshot object stores;
- restore-time cleanup of incomplete metadata, non-ready segments, and expired leases.

PR #1465 added task-manager serialization/deserialization. PR #1381 designed an etcd snapshot object store so that snapshot data can also use the HA etcd deployment.

Snapshot is still periodic full-state persistence. It cannot recover mutations that happen after the latest snapshot. OpLog is therefore needed to reduce that loss window and let the standby catch up before promotion.

### RFC #1200: Initial OpLog Hot Standby

RFC #1200 proposed using etcd as a reliable intermediate component for primary-standby OpLog synchronization:

- On the primary, `OpLogManager` assigns a globally increasing `sequence_id` and creates `OpLogEntry` records for metadata mutations.
- `EtcdOpLogStore` persists each entry to etcd, for example under `/oplog/{cluster}/{sequence_id}`.
- `/oplog/{cluster}/latest` records the latest sequence.
- On the standby, `OpLogWatcher` receives real-time changes through etcd watch and reads historical entries during startup or reconnect.
- `OpLogApplier` checks ordering and applies entries to standby metadata.
- `HotStandbyService` manages standby lifecycle and promotion.

The main design points were correctness and RPO/RTO: global sequence ordering, etcd persistence, etcd watch, snapshot sequence recording, and OpLog cleanup.

### PR #1451 and PR #1515: Early Implementation

PR #1451 wrapped the basic etcd interface needed by Master HA.

PR #1515 implemented the early hot-standby and OpLog interface:

- `EtcdOpLogStore`
- `OpLogManager`
- `OpLogWatcher`
- `OpLogApplier`
- `HotStandbyService`
- `StandbyStateMachine`
- OpLog types such as `PUT_END`, `PUT_REVOKE`, and `REMOVE`
- unit tests for standby metadata store behavior, gap handling, watch/reconnect, and promotion

This moved the OpLog design from RFC into a runnable implementation. However, the early code was tightly coupled to etcd, and the Mooncake Store metadata schema, tenant model, segment lifecycle, and promotion runtime have changed since then.

### RFC #1650: OpLog Store Abstraction

RFC #1650 addressed the etcd coupling in the early OpLog subsystem. Several upper-layer components directly held or constructed `EtcdOpLogStore`, while the old watcher mixed watch, reconnect, deserialization, and apply orchestration in one class.

The RFC proposed:

- an `OpLogStore` interface for write, read, read-since, latest-sequence, snapshot-sequence, and cleanup operations;
- an `OpLogChangeNotifier` interface for backend-specific notification;
- an `OpLogReplicator` that replaces the monolithic etcd-centric watcher;
- shared OpLog serialization;
- centralized store creation through a factory;
- mock stores/notifiers for deterministic unit tests.

That RFC intentionally kept behavior unchanged. It extracted interfaces and clarified boundaries, but did not change the legacy per-entry OpLog layout or logical entry semantics.

### HA Runtime Refactoring

PR #1678 introduced an HA backend abstraction and moved leader coordination from the legacy etcd helper path to `LeaderCoordinator`. It also kept `OpLogStore` and `SnapshotStore` as backend boundaries for follow-up work. PR #1722 added a Redis leader coordinator, proving that the leadership path can be backend-independent.

RFC #1648 discussed Kubernetes Lease-based HA. That work is about leader coordination and master-view discovery. It does not by itself provide OpLog durability or replay semantics.

PR #1777 split the always-on Master process runtime from the leader-only serving runtime, introduced standby-controller composition, and made snapshot bootstrap plus OpLog following part of the standby runtime. This is the runtime foundation needed for switching the standby between legacy per-entry replay and batch-record replay.

### PR #2331: Current Foundation

PR #2331 (`[Store] add HA OpLog replication and tenant-aware standby promotion`) rebased and adapted the earlier HA hot-standby work to current main. The key changes include:

- HA OpLog configuration: `oplog_store_type`, `oplog_store_root_dir`, and `oplog_poll_interval_ms`;
- primary-side OpLog publishing for Store metadata mutation paths;
- durable OpLog guarding for HA-enabled mutations where dropping the update would make primary and standby diverge;
- `SEGMENT_MOUNT`, `SEGMENT_UNMOUNT`, and `SEGMENT_UPDATE` OpLog entries for standby segment registry replay;
- `tenant_id` in `OpLogEntry`, tenant-aware metadata store APIs, and tenant-aware standby snapshot/restore paths;
- backward-compatible OpLog deserialization when old entries do not contain `tenant_id`;
- standby snapshot export and promoted-primary restore for object metadata, group membership, segment state, and the OpLog sequence baseline;
- tests for OpLog apply, standby promotion, tenant isolation, segment lifecycle replay, and MasterService HA mutation behavior.

After #2331, current main has a much more complete OpLog HA baseline. The remaining issue is the storage and write model: the legacy path still stores one backend key per logical `OpLogEntry`, and `/latest` remains the latest-sequence pointer. `EtcdOpLogStore::WriteOpLog(entry, true)` already uses an internal batch writer, so the business thread is not directly writing each single key to etcd. But from the Master mutation path, synchronous waiting can still happen while holding `snapshot_mutex_` in shared mode and a metadata shard lock in exclusive mode.

## Motivation

The current #2331-based HA OpLog path covers many business cases, but the write protocol still has several structural problems.

### 1. Lock-held Durable Wait

Several paths can wait for durable OpLog persistence while holding `snapshot_mutex_` and a metadata shard lock:

- `Remove()`
- `BatchRemove()`
- `RemoveAll()`
- `BatchReplicaClear()`
- `BatchEvict()`
- `NoFBatchEvict()`
- `BatchEvictDiskReplica()`
- stale-handle cleanup
- expired processing/replication cleanup
- segment lifecycle updates
- offload/promotion success paths

The current sync write path enqueues entries into an internal pending batch and waits for the flush result. etcd latency, batch-flush delay, retry, or sync timeout can therefore block unrelated metadata mutations in the same shard.

### 2. Per-entry Backend Key Amplification

The legacy layout is:

```text
/oplog/{cluster}/{sequence_id}
/oplog/{cluster}/latest
/oplog/{cluster}/snapshot/{snapshot_id}/sequence_id
```

Each logical OpLog entry becomes one backend key. Under high mutation concurrency, the backend must handle many keys, watch events, MVCC items, and compaction items. Internal group commit reduces the number of flush calls, but it does not reduce the number of persisted keys.

### 3. Split Safety Boundary

In the legacy model, an entry key means the entry exists, while `/latest` means the primary has advanced the latest sequence pointer. For standby apply and promotion, the important concept is a verified, continuous, no-gap durable boundary.

Batch-record mode should make this explicit. A batch record and the durable boundary should advance together through a CAS/txn protocol, and standby/promotion should trust only that durable boundary.

### 4. Need for a Post-#2331 Evolution Path

#2331 adapted the legacy HA OpLog implementation to current main and added tenant, segment, and promotion compatibility. The next step should not rewrite those semantics. It should keep:

- current logical `OpLogEntry` semantics;
- current `PUT_END`, `REMOVE`, and `SEGMENT_*` apply behavior;
- tenant-aware metadata;
- standby snapshot export and promoted-primary restore;
- legacy compatibility during upgrade.

The part to replace is the underlying durable write and replay protocol.

## Goals

1. Replace new per-entry backend keys with **batch records**. One batch record is one backend KV value containing multiple logical `OpLogEntry` records over a continuous sequence range.
2. Use `{batch_id, last_seq}` as the **durable prefix** and the only safe boundary for callbacks, standby apply, and promotion.
3. Move OpLog admission, sequence assignment, batching, retry, and durable callbacks out of metadata shard lock hold time, or reduce the lock-held path to a short enqueue.
4. Keep logical sequence globally monotonic and continuous.
5. Keep the existing logical meaning of `OpLogEntry`.
6. Support mixed reading of legacy per-entry OpLog and new batch-record OpLog in the same cluster, without double-writing.
7. Require txn/CAS for the first version. Non-txn backends remain a future extension and must not be claimed as HA-correct in this version.
8. On backend failure, fail closed: stop new write admission, do not skip the stuck batch, do not report success before durability, and retry until backend recovery or leadership change.

## Non-Goals

1. Do not introduce a new delta OpLog schema in the first version. Batch entries still use the current full `OpLogEntry` format.
2. Do not double-write legacy per-entry OpLog and batch-record OpLog.
3. Do not require old standbys to read new batch records. Operators must upgrade standby/readers first, then switch primary to batch-record mode.
4. Do not claim HA correctness for non-txn backends in the first version.
5. Do not put leadership lease/watch/election semantics into the minimal KV backend interface.
6. Do not fully redesign `UpsertStart` in this RFC. That path needs a separate design because its semantics are more complex.

## Proposal

### 1. Architecture

Introduce three layers:

```cpp
class OrderedOpLogWriter;
class OpLogBatchStorage;
class HaKvBackend;
```

#### OrderedOpLogWriter

`OrderedOpLogWriter` is the only OpLog admission, sequence assignment, batching, retry, and callback-dispatch entry point in a serving primary.

Business code uses:

1. `Reserve()` before metadata mutation.
2. `Commit()` inside the metadata shard lock to attach the final `OpLogEntry` and durable callback.
3. `Abort()` if the precondition check fails or no OpLog is needed.

The writer owns:

- writer health and `accepting`;
- open waiting-batch slot accounting;
- logical sequence assignment;
- work-conserving batch seal;
- single in-flight writing batch;
- backend txn/CAS write;
- durable prefix advancement;
- retry and unknown-outcome handling;
- callback dispatch in logical sequence order.

#### OpLogBatchStorage

`OpLogBatchStorage` is a layout/storage helper between the writer and the KV backend. It owns:

- batch-record key construction;
- durable-prefix key construction;
- batch-record encoding/decoding;
- durable-prefix encoding/decoding;
- checksum validation;
- etcd txn/CAS write;
- range reading of batch records;
- legacy `/latest` read for cutover initialization.

It does not own admission, sequence assignment, retry policy, callback dispatch, or Master business semantics.

#### HaKvBackend

`HaKvBackend` is a reusable HA KV adapter layer. It provides only generic KV/range/transaction capabilities. It does not know about OpLog, durable prefixes, leadership, snapshot, or metadata semantics.

The first production implementation should be an etcd txn/CAS backend. Future leadership reuse should use sidecar interfaces such as `HaKvLeaseBackend` or `HaKvWatchBackend`, instead of adding lease/watch/election methods to the minimal KV interface.

### 2. Core Data Structures

```cpp
struct DurablePrefix {
    uint64_t batch_id{0};
    uint64_t last_seq{0};
};

struct OpLogBatchRecord {
    uint32_t schema_version{1};
    uint64_t batch_id{0};
    uint64_t first_seq{0};
    uint64_t last_seq{0};
    std::vector<OpLogEntry> entries;
    uint32_t checksum{0};
};
```

`DurablePrefix` means every logical entry in `[1, last_seq]` is durable and safe to apply/promote from. `batch_id` identifies the physical batch record that contains `last_seq`.

`OpLogBatchRecord` must satisfy:

- `entries` is non-empty;
- entries are sorted by increasing `sequence_id`;
- `first_seq == entries.front().sequence_id`;
- `last_seq == entries.back().sequence_id`;
- sequence IDs are continuous inside a batch;
- sequence IDs are continuous across batches;
- `batch_id` is monotonic;
- checksum covers the encoded batch content except the checksum field itself.

### 3. Minimal KV Backend Interface

```cpp
struct KvPair {
    std::string key;
    std::string value;
};

enum class KvCompareKind {
    kValueEquals,
    kKeyNotExists,
};

struct KvCompare {
    std::string key;
    KvCompareKind kind{KvCompareKind::kValueEquals};
    std::string expected_value;
};

struct KvTxn {
    std::vector<KvCompare> compares;
    std::vector<KvPair> puts;
};

class HaKvBackend {
public:
    virtual ~HaKvBackend() = default;

    virtual ErrorCode Get(std::string_view key, std::string& value) = 0;
    virtual ErrorCode Put(std::string_view key, std::string_view value) = 0;
    virtual ErrorCode Range(std::string_view begin_key,
                            std::string_view end_key,
                            size_t limit,
                            std::vector<KvPair>& kvs) = 0;
    virtual bool SupportsTxn() const = 0;
    virtual ErrorCode Txn(const KvTxn& txn) = 0;
};
```

Batch-record HA initialization must check `SupportsTxn()`. If the backend does not support txn/CAS, the first version should fail initialization or return unsupported.

### 4. Storage Layout

Keep legacy keys unchanged:

```text
/oplog/{cluster}/{seq_padded}
/oplog/{cluster}/latest
/oplog/{cluster}/snapshot/{snapshot_id}/sequence_id
```

Add batch-record keys under the same prefix:

```text
/oplog/{cluster}/batches/{batch_id_padded}
/oplog/{cluster}/durable_prefix
```

Use fixed-width decimal encoding for `batch_id`, preferably the same 20-digit padding used by legacy sequence keys. This keeps lexicographic range scan order equal to numeric order.

Since the new keys remain under `/oplog/{cluster}/`, readers must explicitly filter:

```text
/latest
/snapshot/
/batches/
/durable_prefix
```

### 5. Durable Prefix Value

`/oplog/{cluster}/durable_prefix` should contain at least:

```text
schema_version=<version>
batch_id=<id>
last_seq=<seq>
checksum=<optional>
```

The actual encoding can use an existing project serialization format, but it must include `schema_version` so later versions can add leader epoch, term, primary id, or layout version.

Semantics:

- `[1, last_seq]` is the only safe apply/promotion boundary.
- `batch_id` locates the batch containing `last_seq`.
- standby and promotion must use durable prefix, not `/latest`, as the safety boundary.
- batch-record mode does not write new `/latest` values.
- legacy `/latest` is used only for cutover initialization and legacy reader compatibility.

### 6. Cutover and Legacy Compatibility

Upgrade order:

```text
1. Upgrade standby/readers so they can read both legacy per-entry OpLog and batch-record OpLog.
2. Switch the primary to batch-record writer.
3. After the switch, the primary stops writing new legacy per-entry keys and stops updating /latest.
```

If a batch-record writer starts and durable prefix does not exist:

```text
legacy_latest = Read(/oplog/{cluster}/latest)
durable_prefix = { batch_id = 0, last_seq = legacy_latest }
next_sequence = legacy_latest + 1
next_batch_id = 1
```

Create durable prefix with CAS/create-if-absent:

```text
compare durable_prefix key does not exist
put durable_prefix = { batch_id = 0, last_seq = legacy_latest }
```

If the compare fails, another initializer has created the durable prefix. The current writer must read the existing prefix and use it as the source of truth.

This design intentionally trusts `/latest` during cutover:

- do not scan legacy max sequence;
- do not check for legacy keys with `seq > latest`;
- if such keys exist, they are treated as outside the new durable boundary.

This is an operational tradeoff. It assumes the old write path has been stopped during cutover, or that `/latest` is the accepted legacy safety boundary.

### 7. Writer Admission Protocol

Add:

```text
oplog_batch_max_entries = 1024
```

This limits the number of entries in the current open waiting batch. It is an admission/backpressure limit, not a target batch size.

Business code uses:

```cpp
tl::expected<Reservation, ErrorCode> Reserve();

tl::expected<PendingHandle, ErrorCode> Commit(Reservation&& reservation,
                                              OpLogEntry entry,
                                              DurableCallback callback);

void Abort(Reservation&& reservation);
```

#### Reserve

`Reserve()` runs before metadata mutation:

```text
if (!accepting) return OPLOG_UNAVAILABLE;
if (open_waiting_reserved_or_committed >= oplog_batch_max_entries)
    return QUEUE_FULL;
++open_waiting_reserved_or_committed;
return Reservation{};
```

`Reserve()` does not allocate a sequence ID. Therefore, `Abort()` does not create a gap.

#### Commit

`Commit()` runs inside the metadata shard lock, but must stay short:

- consume an existing reservation;
- assign the next continuous `sequence_id`;
- fill the entry;
- append it to the waiting queue;
- attach the durable callback;
- notify the writer;
- do no backend I/O;
- do not wait for queue space;
- do not serialize the full batch record.

If the caller already has a reservation and the entry uses a known serializable OpLog format, normal `Commit()` should not fail.

#### Abort

`Abort()` releases a reservation if the precondition check fails or no OpLog is needed:

- release the slot;
- do not allocate sequence;
- do not create a gap;
- do not trigger callbacks.

### 8. Batch Seal and Flush Policy

The first version uses a **work-conserving single in-flight writer**:

```text
1. If writing_batch is idle and waiting_queue has entries, seal the current open waiting batch immediately.
2. Even one entry is enough to write a batch.
3. If writing_batch is in progress, business threads can keep reserve/commit into a new open waiting batch.
4. After writing_batch succeeds:
   - if waiting batch has entries, seal and write the next batch immediately;
   - otherwise sleep until commit notification.
```

Implications:

- `oplog_batch_max_entries` is a backpressure limit, not a flush threshold.
- Low traffic may produce one-entry batches.
- High traffic or slower backend writes will naturally coalesce entries.
- The design optimizes latency first and does not intentionally wait to fill a batch.

Seal does not wait for reservations that have not yet committed or aborted. Already committed entries can be drained and sealed. Slow business threads keep their reserved slots and their later commits go into a later seal.

### 9. Sequence and Batch ID

Logical sequence is assigned in `Commit()`:

```text
next_sequence starts from durable_prefix.last_seq + 1
Commit() assigns current next_sequence
next_sequence++
```

Guarantees:

- `Abort()` creates no gap;
- committed entries are continuous;
- entries inside a batch are continuous;
- batches are continuous;
- standby strict no-gap validation can hold.

`batch_id` is assigned at seal time:

```text
next_batch_id starts from durable_prefix.batch_id + 1
seal waiting batch assigns current next_batch_id
next_batch_id++
```

`batch_id` does not replace logical sequence. It locates batch records, describes the physical durable prefix position, supports standby range reads, and lets promotion verify batch continuity.

### 10. Backend txn/CAS Protocol

For one writing batch:

```text
expected_prefix = current durable prefix before this batch
new_prefix = { batch_id = batch.batch_id, last_seq = batch.last_seq }

txn:
  compare /oplog/{cluster}/durable_prefix == Encode(expected_prefix)
  put /oplog/{cluster}/batches/{batch_id_padded} = Encode(batch)
  put /oplog/{cluster}/durable_prefix = Encode(new_prefix)
```

After txn success:

- batch record is readable;
- durable prefix points to this batch;
- callbacks can be dispatched;
- standby can apply through `new_prefix.last_seq`.

On txn failure or network timeout:

1. Re-read durable prefix.
2. If durable prefix equals `new_prefix` and the batch record is readable and valid, treat the previous unknown outcome as success.
3. If durable prefix still equals `expected_prefix`, retry the txn.
4. If durable prefix is older than expected, or newer but not equal to this batch's `new_prefix`, treat it as stale writer / concurrent primary / data anomaly. Keep writer unhealthy and do not skip.

Durable prefix must be monotonic. Exact-value compare prevents overwrite and rollback.

### 11. Backend Failure Policy

The first version uses:

> fail-closed admission + indefinite retry + success auto-recover

On the first backend txn or durable-prefix advancement failure:

```text
keep writing_batch in place
accepting=false
new Reserve() fails fast
do not dispatch success callbacks
do not write later batches
retry current writing_batch indefinitely
```

For committed but not durable mutations:

- the writer does not actively return an error;
- RPC/finalize waiters remain pending;
- if retry eventually succeeds, callbacks complete normally;
- if leadership changes or the service shuts down, upper-layer RPC deadline, connection close, or shutdown handling ends client waiting.

After a successful retry:

```text
durable prefix advances
callbacks are dispatched in sequence order
writing_batch is cleared
accepting=true
writer immediately checks the waiting batch
```

Later batches must never pass a stuck batch. Passing it would create a durable-prefix gap and break standby/promotion safety.

### 12. Durable Callback and Finalize

Callbacks fire only after durable prefix has advanced:

```text
batch record readable
batch checksum valid
durable_prefix points to this batch
=> callbacks in this batch may be dispatched in logical sequence order
```

Callback must not fire just because the batch record was written, and must not depend on legacy `/latest`.

The writer thread should only enqueue callback tasks. It should not perform heavy business logic, resource release, quota accounting, or metadata relocking.

Remove/evict/clear finalize callbacks may perform:

- `EraseMetadata()`;
- replica erase/finalize;
- quota release;
- local disk usage release;
- cache total accounting;
- `disk_object_count` updates;
- group-member unregister;
- release of moved-out replica/resource state;
- RPC waiter wakeup.

The callback context must include enough information to finalize safely:

- tenant id;
- object key;
- sequence id;
- operation kind;
- affected replica ids;
- expected replica status;
- quota/resource release mode.

Finalize must re-enter the shard lock and verify that current metadata still matches the pending mutation. If the object has been recreated by a higher-sequence write, the callback must not delete the new metadata.

### 13. Visibility Semantics

#### Operations That Reduce Readability

Operations that reduce readable replica/object state may make the state invisible on the primary before durability:

- `Remove()`
- `BatchRemove()`
- `RemoveAll()`
- `BatchReplicaClear()`
- `BatchEvict()`
- `NoFBatchEvict()`
- `BatchEvictDiskReplica()`
- stale-handle cleanup
- expired processing/replication cleanup

Target flow:

```text
1. Reserve().
2. Acquire snapshot_mutex_ shared lock and metadata shard lock.
3. Check object, lease, pin, refcnt, task, and other preconditions.
4. Build the current-format OpLog entry.
5. Mark affected replicas as ReplicaStatus::REMOVED, or otherwise stop returning them as COMPLETE.
6. Commit(reservation, entry, durable callback).
7. Release locks.
8. After durable callback, perform real erase/finalize and resource release.
```

Resource release must not happen before durable callback. This includes buffer/resource reuse, quota release, local disk usage release, cache accounting, disk object count, and group membership cleanup.

`ReplicaStatus::REMOVED` means:

- logically unreadable;
- not returned by `GetReplicaList()`;
- resources are not finalized yet;
- physical erase/release happens after durable callback.

If other `COMPLETE` replicas remain, reads return those readable replicas. If all readable replicas are removed, reads return `OBJECT_NOT_FOUND`.

#### Operations That Increase Readability

Operations that add readable replica/object state may remain visible on the primary before durability. If the primary crashes before durability, the new primary may lose that added readable state.

This applies to:

- `PutEnd`
- `AddReplica`
- `NotifyOffloadSuccess()` fallback
- `NotifyPromotionSuccess()`
- copy/move target complete
- other paths that mark replicas `COMPLETE`

This keeps the first version close to current behavior and avoids introducing a pending-visible state for every readability-increasing path.

#### No New PendingVisibilityState

The first version does not add a new object-level `PendingVisibilityState`. Read-path distinction should use existing `ReplicaStatus`:

```text
Has COMPLETE replica
    => OK, return COMPLETE descriptors
No COMPLETE replica, all remaining replicas are REMOVED or metadata only represents removed state
    => OBJECT_NOT_FOUND
No COMPLETE replica, but has PROCESSING / INITIALIZED / other in-progress status
    => REPLICA_IS_NOT_READY
Metadata entry absent / invalid
    => OBJECT_NOT_FOUND
```

### 14. Standby Apply

The first version uses polling, not batch-key watch:

```text
1. Periodically read /oplog/{cluster}/durable_prefix.
2. If durable_prefix does not exist, continue using the legacy path.
3. If durable_prefix exists, treat it as the only safety boundary.
4. Range-read /oplog/{cluster}/batches/ after the last applied batch id.
5. Validate schema, checksum, batch_id continuity, and sequence continuity.
6. Expand entries.
7. Apply entries in logical sequence order.
```

Legacy and batch path transition:

- If durable prefix does not exist, standby uses legacy `/latest` plus per-entry keys.
- If durable prefix exists and `batch_id == 0`, it is a cutover boundary. Legacy `[1,last_seq]` is the safe boundary.
- If durable prefix exists and `batch_id > 0`, standby must first apply legacy entries through the cutover `last_seq`, then read batch records.

Batch-record mode must be strict no-gap. If durable prefix is unreadable, a referenced batch is unreadable, batch IDs are discontinuous, `first_seq != local_last_applied_seq + 1`, entries are discontinuous, checksum fails, schema version is unsupported, or deserialization fails, standby must become unhealthy/fail-closed. It must not skip the gap, and it must not be promoted.

Batch entries still use the current `OpLogEntry`:

- `PUT_END`: apply full metadata post-state;
- `REMOVE`: delete object metadata;
- `SEGMENT_*`: update standby segment registry;
- `tenant_id`: keep #2331 tenant-aware semantics, defaulting missing legacy values to `default`.

As follow-up hardening, `PUT_END` apply should consistently use object `last_sequence_id` stale checks. Strict no-gap reduces reordering risk, but stale checks still help against duplicates, retries, and legacy/batch mixed anomalies.

### 15. Promotion

Before promotion, standby must catch up to durable prefix:

```text
local_last_applied_seq == durable_prefix.last_seq
```

Promotion must fail closed if:

- durable prefix cannot be read;
- the batch referenced by durable prefix cannot be read;
- batch checksum or schema validation fails;
- batch_id or sequence gap remains unresolved;
- local applied sequence is behind durable prefix and cannot catch up;
- batch-record mode has any unresolved gap.

In batch-record mode, `/latest` is not a promotion safety boundary.

### 16. Relation to Current Code

The current write entry points should be gradually replaced or routed through the new writer:

- `AppendOpLogAndNotify()`
- `AppendOpLogAndNotifyDurable()`
- `AppendOpLogAndNotifyDurableOrAbort()`
- `PersistOpLogEntryWithSyncRetries()`
- `AppendOrPersistOrEnqueue()`
- `PersistRemoveForHA()`
- `PersistSegmentOpForHAOrEnqueue()`

Legacy `EtcdOpLogStore` can still be used for:

- legacy per-entry reads;
- reading `/latest` during cutover;
- legacy standby path when durable prefix does not exist;
- compatibility with old snapshot-boundary reads.

The batch-record writer should not write new entries through `EtcdOpLogStore::WriteOpLog()`.

## Implementation Plan

### Phase 1: Batch-record Writer Foundation

- Add `oplog_batch_max_entries`, default 1024.
- Define `DurablePrefix` and `OpLogBatchRecord`.
- Define `ha/kv/HaKvBackend` and an etcd txn/CAS backend.
- Define `OpLogBatchStorage`.
- Implement batch key, durable-prefix key, and value codecs.
- Implement checksum and schema-version validation.
- Implement `OrderedOpLogWriter`:
  - `Reserve / Commit / Abort`;
  - sequence assignment in `Commit()`;
  - work-conserving seal;
  - single in-flight writing batch;
  - `accepting=false` after first backend failure;
  - indefinite retry for stuck batch;
  - `accepting=true` after retry success;
  - callback dispatch in sequence order.
- Do not write new `/latest` values in batch-record mode.

### Phase 2: Standby Dual-format Reader

- Use legacy path when durable prefix does not exist.
- Enter batch-record path when durable prefix exists.
- If standby is behind the cutover legacy latest, catch up through legacy per-entry replay first.
- Poll durable prefix.
- Range-read batches.
- Expand and apply batch entries.
- Disable normal gap skip in batch-record mode.
- Require final catch-up to durable prefix before promotion.

### Phase 3: Basic Write-path Integration

Priority paths:

- `PutEnd` / `AddReplica`
- segment lifecycle
- `Remove` / `BatchRemove` / `RemoveAll`
- `BatchReplicaClear`
- `BatchEvict` / `NoFBatchEvict` / `BatchEvictDiskReplica`
- offload success
- promotion success

The initial integration can keep most current visibility behavior and only replace the durable write entry point. This keeps the first PR reviewable and avoids changing all resource-release semantics at once.

### Phase 4: Remove/Evict Logical Remove and Durable Finalize

- In the short lock-held path, mark affected replicas as `REMOVED`.
- Release locks after `Commit()`.
- Perform real erase/finalize in the durable callback.
- Adjust read path using existing `ReplicaStatus`:
  - at least one `COMPLETE`: OK;
  - all readable replicas removed or absent: `OBJECT_NOT_FOUND`;
  - only in-progress replicas: `REPLICA_IS_NOT_READY`.

### Phase 5: Health and Operations

- Expose writer `accepting=false` / unhealthy state to runtime/admin state.
- Fail fast for new mutating requests.
- Let leadership/supervisor decide whether to stop serving or trigger failover.
- Add metrics for durable prefix lag, retry count, stuck batch age, queue size, callback latency, and admission failures.
- Later add durable-prefix watch, byte limits, non-txn backend exploration, and `ha/kv` lease/watch sidecar interfaces.

### Phase 6: UpsertStart Follow-up

`UpsertStart` has more complex semantics:

- same-size in-place rewrite;
- size-changing replacement;
- old `PROCESSING` preemption;
- quota replacement;
- discarded replicas.

This RFC does not make `UpsertStart` a blocker for the first batch-record writer version. A separate design should define old-readable visibility, new-readable visibility before durability, failure handling, quota/resource handling, and compatibility with current-format full-state entries.

### Phase 7: Delta Schema Follow-up

Delta schema is not part of the first version. Future work can introduce:

- `OBJECT_MARK_REMOVED`
- `REPLICA_MARK_REMOVED`
- `OBJECT_CLEAR_REPLICAS`
- `OBJECT_MARK_UNAVAILABLE`
- `REPLICA_ADD`
- `REPLICA_UPDATE`
- `REPLICA_MARK_COMPLETE`

Apply rules, idempotency, legacy mixed apply, snapshot relation, and compaction relation should be designed separately.

## Testing Strategy

### Storage and Codec

- batch key padding preserves numeric range order;
- durable prefix encode/decode works;
- batch record encode/decode works;
- checksum detects corrupted values;
- unsupported schema version fails closed;
- reader filters `/latest`, `/snapshot/`, `/batches/`, and `/durable_prefix` correctly.

### txn/CAS

- txn puts batch record and durable prefix together;
- expected durable-prefix compare advances prefix on success;
- compare failure does not trigger callback;
- unknown outcome retry recognizes success when prefix already advanced and batch validates;
- durable prefix cannot roll back;
- non-txn backend returns unsupported for first-version batch-record HA.

### Writer Admission

- `Reserve()` does not allocate sequence;
- `Abort()` creates no gap;
- `Commit()` assigns continuous sequence;
- `Reserve()` limits open waiting batch only, not writing batch;
- reaching `oplog_batch_max_entries` makes new `Reserve()` fail fast;
- seal releases committed slots so new reservations can enter the next open waiting batch;
- `Commit()` does no backend I/O, no waiting, and no full batch serialization.

### Writer Flush / Retry / Callback

- an idle writer writes even a one-entry batch immediately;
- single in-flight writer prevents later batches from passing a stuck batch;
- first backend failure sets `accepting=false`;
- new `Reserve()` fails while `accepting=false`;
- already reserved callers may still commit or abort;
- stuck batch retries indefinitely;
- retry success restores `accepting=true`;
- callback does not fire before durable prefix advancement;
- callbacks dispatch in logical sequence order;
- slow callback execution does not block the next backend write.

### Legacy Compatibility

- durable prefix absent means standby uses legacy path;
- cutover initializes `{batch_id=0,last_seq=latest}` from `/latest`;
- cutover does not scan max sequence;
- standby behind legacy latest catches up with legacy per-entry replay first;
- first batch satisfies `first_seq == legacy_latest + 1`;
- batch-record mode does not write new `/latest`.

### Standby and Promotion

- poll durable prefix and range-read batches;
- validate batch_id continuity;
- validate first_seq/last_seq continuity;
- validate entry sequence continuity;
- checksum/schema/deserialize failures fail closed;
- batch-record mode does not use normal gap skip;
- promotion requires catch-up to durable prefix;
- promotion fails closed if durable prefix cannot be reached.

### MasterService Integration

- `PutEnd` / `AddReplica` remain visible on primary according to current semantics;
- `Remove` / `BatchRemove` / `RemoveAll` keep durable guard correctness;
- `BatchReplicaClear`, `BatchEvict`, `NoFBatchEvict`, and `BatchEvictDiskReplica` preserve pin/refcnt/lease preconditions;
- segment mount/unmount/update replay into standby segment registry;
- tenant-aware keys remain isolated in standby metadata;
- promotion restore preserves tenant, `group_id`, `data_type`, and segment registry state;
- durable finalize does not delete metadata recreated by a higher sequence.

## Key Design Points Summary

1. **Build on #2331**: keep current OpLog semantics, tenant-aware metadata, segment lifecycle replay, and promotion handoff.
2. **Batch record is the new durable unit**: reduce per-entry backend key amplification.
3. **Durable prefix is the only safety boundary**: callbacks, standby apply, and promotion use `{batch_id,last_seq}`.
4. **Sequence is assigned in Commit**: `Reserve()` does not allocate sequence and `Abort()` creates no gap.
5. **Work-conserving writer**: no artificial wait to fill a batch; high concurrency naturally coalesces entries.
6. **Fail closed**: first backend failure stops new admission, the stuck batch is not skipped, and retry success recovers automatically.
7. **Legacy compatible without double-write**: new readers support both formats; new primary stops writing legacy per-entry keys after cutover.
8. **First version requires txn/CAS**: non-txn backends are future work and are not HA-correct in this RFC.
9. **No delta schema in the first version**: batch entries still use current full `OpLogEntry`.
10. **Strict no-gap in batch-record mode**: standby cannot skip batch gaps, and promotion must catch up to durable prefix.

### Before submitting a new issue...

- [ ] Make sure you already searched for relevant issues and read the [documentation](https://kvcache-ai.github.io/Mooncake/)