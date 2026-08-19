# Hidden behavioral suite — layout

A test file's PATH declares its audience; the grader positively selects
directories per problem (`suites` in `src/problems/p-<id>.ts`):

- `core/` — workspace-shape-agnostic; runs for EVERY problem.
- `gold/` — exercises features/IR fields that exist only on gold-lineage
  workspaces.
- `base/` — base-family ground-truth twins (also env-gated at runtime).
- `problems/<id>/` — tests that exist ONLY because that problem exists
  (harnesses, discriminators). Sibling ids sharing a workspace share a dir.
- `_lib/` — shared helpers (never collected).
- `conftest.py`, `fixtures/` — shared root; conftest puts this directory on
  sys.path so any subdirectory can `from _lib.<mod> import ...`.

A problem's GRADED (f2p) tests are not necessarily in its `problems/<id>/`
dir — shared behavioral tests in `core/`/`gold/`/`base/` often double as
truth anchors. The authoritative map is the problem's f2p list in
`src/problems/p-<id>.ts`, whose keys are full node ids
(`core/test_oracle.py::test_x`): the key itself names the defining file,
and the grader validates at grade time that each file exists and sits
inside the problem's suites. A renamed or missing f2p test additionally
fails closed (missing-f2p injection), so a mapping error can never
silently misgrade.
