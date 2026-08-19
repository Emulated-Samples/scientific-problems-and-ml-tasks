# Batch OpLog wire format

Normative encoding contract for the batch-record OpLog codec
(`ha/oplog/oplog_batch_codec.h`, `ha/oplog/oplog_batch_types.h`,
`ha/oplog/oplog_types.h`). Interoperability with existing tooling requires
byte-exact conformance.

## 1. Canonical JSON serialization

Every byte emitted or checksummed uses JsonCpp compact form
(`Json::StreamWriterBuilder` with `builder["indentation"] = ""`): a single
line, no spaces after `,` or `:`, no trailing newline, unsigned integers as
plain decimal, strings double-quoted with JsonCpp's default escaping.

Payload bytes are carried as **base64** (standard alphabet `A–Za–z0–9+/`,
`=` padding to a multiple of 4; `utils/base64.h` provides Encode/Decode).
Example: the 7-byte payload `"\0binary"` encodes to `"AGJpbmFyeQ=="`; the
empty payload encodes to `""`.

## 2. Error-reporting convention

All decode/validate functions take `std::string* reason` (nullable, default
null):

- On failure: return `false` and, when `reason` is non-null, set it to a
  non-empty human-readable string (the normative wordings are listed below
  with each rule).
- On success: `reason` ends up empty.
- Passing `reason == nullptr` must be safe everywhere.
- Decoders must never throw on malformed or type-confused input.

## 3. DurablePrefix codec

`EncodeDurablePrefix` produces a canonical-compact JSON **object**:

| key              | JSON type | value |
|------------------|-----------|-------|
| `schema_version` | uint32    | always `kDurablePrefixSchemaVersion` (1) |
| `batch_id`       | uint64    | `prefix.batch_id` |
| `last_seq`       | uint64    | `prefix.last_seq` |

`DecodeDurablePrefix`, in order (each failure: `false`, non-empty reason,
no throw):

1. Parse as JSON (`"malformed json"` / parser text on failure); root must be
   an object (`"durable prefix must be a JSON object"`).
2. `schema_version` present and `isUInt()`
   (`"missing field: schema_version"` / `"field must be uint32:
   schema_version"`), and equal to 1
   (`"unsupported durable prefix schema_version"`).
3. `batch_id`, `last_seq` present and `isUInt64()`
   (`"missing field: <name>"` / `"field must be uint64: <name>"`).
4. Unknown extra keys are tolerated. There is no range validation: the
   all-zero prefix round-trips successfully.

## 4. OpLogBatchRecord codec

`EncodeOpLogBatchRecord` produces a canonical-compact JSON **array of
exactly six positional elements**:

```
[ schema_version, batch_id, first_seq, last_seq, entries, checksum ]
```

- Element 0 is always written as `kOpLogBatchRecordSchemaVersion` (1),
  regardless of the input struct's value.
- `entries` is an array with one element per entry, in order. Each entry is
  a **four-element array** `[op_type_uint, tenant_id, object_key,
  base64(payload)]`.
- `sequence_id`, `timestamp_ms`, `prefix_hash`, and the per-entry checksum
  are **not persisted** (their names must not appear anywhere in the encoded
  bytes). Entry *i* implicitly has `sequence_id = first_seq + i`.
- Element 5 is the batch checksum (Section 5), recomputed by the encoder;
  the input struct's `checksum` field is ignored.

Example (one `PUT_END("tenant","key","value")` entry at seq 10, batch 3):

```
[1,3,10,10,[[1,"tenant","key","dmFsdWU="]],<checksum>]
```

`DecodeOpLogBatchRecord`, in order:

1. Parse JSON (`"malformed json"`); root must be an array of exactly 6
   elements (`"batch record must be a six-element array"`).
2. Types: `[0].isUInt()`, `[1..3].isUInt64()`, `[4].isArray()`,
   `[5].isUInt()` (`"batch record has invalid field types"`) — checked
   before any value access so type confusion never throws.
3. `[0] == 1` (`"unsupported batch record schema_version"`).
4. Sequence/count consistency
   (`"batch sequence range does not match entry count"`) rejects: empty
   `entries`, `first_seq == 0`, `last_seq < first_seq`, or
   `last_seq - first_seq != entries.size() - 1`.
5. Checksum verification (`"batch record checksum mismatch"`): recompute
   per Section 5 from the **parsed** first five elements re-serialized in
   canonical compact form, compare with `[5]`. This runs **before**
   per-entry decoding, so a record with a valid checksum but a bad payload
   surfaces the payload error, not a checksum error.
6. Per-entry decode, index order. For each element of `entries`:
   - exactly 4 elements (`"oplog entry must be a four-element array"`);
   - types uint/string/string/string
     (`"oplog entry has invalid field types"`);
   - `0 < op_type < OP_TYPE_MAX` i.e. 1..7
     (`"oplog entry op_type is outside the enum range"`);
   - canonical base64: `payload = Decode([3])`; if `Encode(payload) != [3]`
     fail with `"oplog entry payload is not canonical base64"` — the reason
     for a payload-encoding failure must contain the substring `base64`;
   - populate: `sequence_id = first_seq + i`, `timestamp_ms = 0`,
     `prefix_hash = 0`, `checksum = ComputeOpLogChecksum(payload)` (so the
     decoded entry passes `VerifyOpLogChecksum`), remaining fields from the
     wire;
   - run `ValidateOpLogBatchEntry` on the result.
7. Run `ValidateOpLogBatchRecordShape` over the fully decoded record.
8. Only on success is the output struct assigned.

## 5. Checksums

- `ComputeOpLogChecksum(payload)` = `XXH32(payload.data(), payload.size(),
  seed 0)` over the raw (binary) payload bytes — not the base64 text.
- Batch checksum = `XXH32(S, 0)` where `S` is the canonical-compact
  serialization of the five-element array `[schema_version, batch_id,
  first_seq, last_seq, entries]` — the record minus the trailing checksum
  element. For the example above,
  `S = [1,3,10,10,[[1,"tenant","key","dmFsdWU="]]]`, byte-exact.
- `VerifyOpLogChecksum(entry)` is true iff
  `ComputeOpLogChecksum(entry.payload) == entry.checksum` AND
  (`entry.prefix_hash == 0` — meaning "no prefix-hash check" — or
  `XXH32(object_key, 0) == prefix_hash`, with the empty key comparing as 0).

## 6. Cluster IDs

`NormalizeAndValidateClusterId(std::string& id)` mutates in place: strip all
trailing `/` characters, then return true if the result is empty (callers
that build keys additionally reject empty) or matches the valid component
rules: non-empty, at most 128 bytes, every byte in `[0-9A-Za-z]`, `_`, `-`,
`.`. A `/` anywhere except trailing makes the id invalid.

`ValidateOpLogBatchClusterId` runs the normalizer on a copy and fails
(`"invalid cluster_id"`) if it returns false or the normalized copy is
empty.

## 7. Validators

`ValidateOpLogEntrySize(entry, reason = nullptr)` — limits are inclusive
(a value exactly at the limit is legal):

- `object_key.size() > kMaxOpLogObjectKeySize` (4096) →
  `"object_key too large: size=<n>"`.
- `payload.size() > kMaxOpLogPayloadSize` (10485760) →
  `"payload too large: size=<n>"`.

`ValidateOpLogBatchEntry(entry, reason = nullptr)`:

1. `op_type` in 1..7 → `"op_type is outside the valid enum range"`.
2. `TenantId(entry.tenant_id).IsValid()` → `"tenant_id is empty or
   invalid"` (an empty raw tenant string promotes to `"default"` and is
   valid).
3. `ValidateOpLogEntrySize`.

`ValidateOpLogBatchRecordShape(batch, reason = nullptr)`, first failure
wins:

1. `schema_version == 1` → `"unsupported schema_version"`.
2. `batch_id != 0` → `"batch_id must be non-zero"`.
3. `first_seq != 0` → `"first_seq must be non-zero"`.
4. `entries` non-empty → `"batch entries must not be empty"`.
5. `first_seq == entries.front().sequence_id` →
   `"first_seq does not match first entry sequence"`.
6. `last_seq == entries.back().sequence_id` →
   `"last_seq does not match last entry sequence"`.
7. `last_seq >= first_seq` and `last_seq - first_seq == entries.size() - 1`
   → `"batch sequence range does not match entry count"`.
8. Per entry *i*: `sequence_id == first_seq + i` →
   `"entry sequences must be contiguous"`, then `ValidateOpLogBatchEntry`
   with its reason propagated.

## 8. Key layout

Let `C` be the normalized cluster id. Every builder normalizes first; on an
invalid or empty id it returns empty string(s) rather than throwing.

- `FormatOpLogBatchId(id)`: decimal, left-zero-padded to
  `kOpLogBatchIdWidth` (20) digits — exactly the width of `UINT64_MAX`, so
  lexicographic order equals numeric order. `1` →
  `"00000000000000000001"`.
- `BuildBatchRecordKey(C, id)` = `"/oplog/" + C + "/batches/" +
  FormatOpLogBatchId(id)`.
- `BuildDurablePrefixKey(C)` = `"/oplog/" + C + "/durable_prefix"`.
- `BuildBatchRecordRange(C, after)` selects ids **strictly greater than**
  `after` as a half-open `[begin_key, end_key)` range:
  `begin_key = P + FormatOpLogBatchId(after + 1)` with
  `P = "/oplog/" + C + "/batches/"`, and `end_key = PrefixEnd(P)` — the
  prefix with its last byte incremented (P ends in `/`, so
  `"/oplog/clusterA/batches0"`). For `after == UINT64_MAX` the range is
  empty: `begin_key == end_key`.
