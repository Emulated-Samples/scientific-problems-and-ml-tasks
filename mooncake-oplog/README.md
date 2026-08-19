# Batch-record HA OpLog

Implement Mooncake Store's batch-record HA OpLog subsystem from its RFC, against an interface contract that has already landed.

Models build the protocol leaves cleanly from the specs, then lose the heaviest-weighted groups to failures they never see: a contract symbol left undefined, a writer that hangs only under the held-out suite, an error path that takes the process down.

## The task

This task challenges the model to implement a complete durability subsystem in a real distributed KV-cache store, from its RFC.

The interface contract has already landed: headers, build wiring, config surface, and docs are final. The implementation has not. The new sources are stubs, the legacy implementation is removed, and the tree does not build until the agent does the work.

Three curated spec documents come with the tree: the design RFC, reviewed implementation notes that record intentional deviations from the RFC, and the codec wire format. From them, the agent builds the whole subsystem:

1. The batch codec, against a fixed byte format
2. Transactional batch storage
3. A fail-closed ordered writer: no sequence gaps, no batch overtaking, in-order durable callbacks
4. The strict no-gap standby reader and applier
5. Master and hot-standby integration, including promotion and snapshot bootstrap

The reference implementation runs to roughly 7,000 lines across 27 production files. The suites that grade the new subsystem are held out of the agent's tree entirely, so the specs are the only guide, and the verifier had to be robust in both directions: partial credit for every subsystem that genuinely works, and no credit from prints, test edits, or a happy path that skips the mutation call sites.

## Verifier design

We grade with the project's own test suites, pinned and restored over the agent's tree at verification. Grading is deterministic and CPU-only.

| What we check | How |
| --- | --- |
| The subsystem works end to end | 463 pinned tests across 16 binaries, weighted by subsystem group, with the HA integration group weighted heaviest |
| Partial work counts | Each group earns its own pass fraction, so a working codec and writer score even when the integration is incomplete |
| Existing behavior kept | The pre-existing master-service and allocator suites run alongside the new ones and must keep passing |
| The tests are the arbiter | Graded binaries are rebuilt from the agent's production code with the original test sources restored, a pinned manifest rejects agent-registered tests, and a canary confirms assertions are live |

## Trace walkthrough

Every run reads the three specs and builds bottom-up: types, codec, storage, the ordered writer, then the services. The codec comes out perfect in all fifteen runs; the spread is decided in the groups no run can execute locally.

### A strong run

1. **Census the contract before trusting the compiler.** Twenty minutes in, the best run printed a thirty-one-symbol missing-definition list for the master service header alone, then greped each name against its sources as it went, a zero for every gap. A test-only eviction hook sat on that list; this run defined it, and its HA binary was one of only seven in the set that linked.
2. **Rebuild the withheld tests yourself.** With the graded suites held out and the test tree off-limits, it wrote its own harness outside the repo: codec and key-layout checks against the spec's worked wire examples, byte for byte down to the checksum, then an end-to-end master, standby, promotion, restore, remount cycle, all green before it closed.
3. **Perfect where it never got to look.** It finished at 0.7444, top of the set, in one hour forty-one minutes and 316 steps: twenty-eight of twenty-eight on the ordered writer, codec, snapshot, and failpoints perfect, fifty-nine of sixty-six on the standby service. The heaviest group still went to zero: its HA binary linked, ran, then terminated when a failed writer initialization brought the process down where the suite expected a graceful failure.

### A failed run

1. **Verify only what is visible.** The most instructive failure ran the pre-existing suites it could see, watched them pass, and closed after an hour and a quarter with a confident final message: implemented end to end, all tests passed. No omissions section.
2. **Miss one symbol in a fixed contract.** It never defined the test-only eviction hook declared in the frozen headers, so the HA integration binary never linked and the sixty-seven-test group it guards recorded zero. Eight of the fifteen runs share exactly this miss.
3. **Ship a writer that hangs when nobody is watching.** Its ordered writer built cleanly, then hung under the graded suite until the runner killed it, zeroing the twenty-eight-test writer group. Four runs hang the same binary.
4. **Be excellent where it counts less.** Its breadth was the cleanest in the set: regression 161 of 161, codec, storage, and snapshot perfect, standby service tied with the winner. Two failures it never witnessed cost it thirty-eight percent of the weight, and it landed at 0.60.

The shipped spread runs from 0.47 to 0.74 and is decided by bookkeeping rather than protocol design: linkage, hangs, and crashes on suites no run ever executes, which is why the run that greped every declared symbol and believed the zeros it printed is the run that won.

## Failure modes

These are the failure modes we saw across the evaluated runs, and most rollouts exhibit several at once.

| Failure mode | What goes wrong |
| --- | --- |
| One missing contract symbol | A single test-only method declared in the fixed headers goes undefined, the HA test binary never links, and the heaviest-weighted group records zero over working code. |
| Concurrency that holds until the graded suite | The ordered writer behaves under the agent's own checks, then hangs under the held-out suite until the runner kills it. |
| Error paths that crash instead of failing closed | Writer initialization, bootstrap, and standby failures should degrade gracefully; runs let them take the test process down, and a crash zeroes every test in the binary. |
| A spec ambiguity answered silently | The spec leaves one question open in the writer's admission lifecycle; a third of the field guessed it one way without ever flagging the choice, and gave up a third of the writer group. |

Every shipped rollout was audited for reward hacking and is clean.

## Running

Replay the reference solution with the same verifier the agent is scored against:

```bash
harbor run -p delivery/scientific-problems-and-ml-tasks/mooncake-oplog --agent oracle -k 1 -o jobs/
```
