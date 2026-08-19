import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../..");
const adapter = fs.readFileSync(path.join(root, "environment/src/index.ts"), "utf8");
const reporting = fs.readFileSync(path.join(root, "environment/src/reporting.ts"), "utf8");
const provision = fs.readFileSync(path.join(root, "environment/provision.sh"), "utf8");
const task = fs.readFileSync(path.join(root, "tasks/from_scratch_pca/task.toml"), "utf8");
const agentImage = fs.readFileSync(path.join(root, "tasks/from_scratch_pca/environment/Dockerfile"), "utf8");
const packagedTest = fs.readFileSync(path.join(root, "tasks/from_scratch_pca/tests/test.sh"), "utf8");
const verifierImage = fs.readFileSync(path.join(root, "tasks/from_scratch_pca/tests/Dockerfile"), "utf8");
const canonicalGrader = fs.readFileSync(path.join(root, "grader/grade.py"), "utf8");
const resourceWatchdog = fs.readFileSync(path.join(root, "grader/resource_watchdog.py"), "utf8");
const submissionRunner = fs.readFileSync(path.join(root, "grader/submission_runner.py"), "utf8");
const generator = fs.readFileSync(path.join(root, "data/generate.py"), "utf8");

function requiredDatasetCounts() {
  const catalogue = generator.match(
    /def grade_specs\(math_key: bytes\).*?return _keyed_specs\(\[(.*?)\], math_key, "grade"\)/s,
  );
  assert.ok(catalogue, "grade_specs catalogue must remain statically auditable");
  const synthetic = (catalogue[1].match(/\bDatasetSpec\s*\(/g) ?? []).length;
  const adapterCount = adapter.match(/const EXPECTED_DATASETS = (\d+);/);
  const packagedCount = packagedTest.match(/len\(truth_paths\) == (\d+)/);
  assert.ok(adapterCount, "adapter must declare its complete dataset count");
  assert.ok(packagedCount, "packaged verifier must declare its complete dataset count");
  return {
    synthetic,
    adapter: Number(adapterCount[1]),
    packaged: Number(packagedCount[1]),
  };
}

test("the observed chromosome bundle is mandatory and provenance pinned", () => {
  const counts = requiredDatasetCounts();
  assert.equal(counts.synthetic, 21);
  assert.equal(counts.adapter, counts.synthetic + 1);
  assert.equal(counts.packaged, counts.adapter);
  for (const category of ["biobank", "crossover", "haploid", "high_rank", "ill_conditioned", "io_wide", "rare_structure", "sample_heavy", "spatial_ld", "spectral_selection", "variable_width", "very_high_rank"]) {
    assert.match(reporting, new RegExp(`"${category}"`));
  }
  assert.match(reporting, /"observed"/);
  assert.match(adapter, /stageObservedDataset\(dataDir\);/);
  assert.match(adapter, /fs\.linkSync\(REAL_VCF, destination\)/);
  assert.match(adapter, /vcf\.dev === fs\.lstatSync\(dataDir\)\.dev/);
  assert.match(adapter, /fs\.copyFileSync\(REAL_VCF, destination, fs\.constants\.COPYFILE_EXCL\)/);
  assert.match(adapter, /derived_sha256/);
  assert.match(adapter, /contract\.spec\?\.derived_bytes !== vcf\.size/);
  assert.match(adapter, /contract\.n_samples !== 500/);
  assert.match(adapter, /contract\.sample_subpop\?\.length !== 500/);
  assert.match(adapter, /contract\.population_names\?\.length/);
  assert.match(adapter, /reviewed observed-cohort bundle is missing or malformed/);
  assert.match(adapter, /randomBytes\(32\)/);
  assert.match(adapter, /--representation-key-file/);
  assert.match(adapter, /--math-key-file/);
  assert.match(adapter, /mathKeyCommitment\(\)/);
  assert.match(adapter, /fs\.rmSync\(representationKey\)/);

  assert.match(provision, /ALL\.chr22\.phase3_shapeit2_mvncall_integrated_v5b/);
  assert.match(provision, /python3\.12 -m venv --clear "\$runtime"/);
  assert.match(provision, /--upgrade "pip==26\.0\.1"/);
  assert.match(provision, /compressed="\$data_root\/\.chr22\.download\.vcf\.gz"/);
  assert.match(provision, /install -d -m 0711 -o root -g root \/opt\/hyperfocal\/pcabench-work/);
  assert.match(provision, /a90c16c4ff2b3196476d506ae13cb3047fae8670163c7c932c4b0239aef3daf5/);
  assert.match(provision, /b4023dc6ee2d62ee89c8d4d347db4d348e65518d66d346574cdae7a4bbd76858/);
  assert.match(provision, /--samples-per-superpop 100 --record-modulus 7/);
  assert.match(provision, /--representation-key-file "\$representation_key"/);
  assert.match(provision, /--math-key-file "\$math_key"/);
  assert.match(provision, /release-v3\.math-key/);
  assert.match(provision, /DERIVATION_VERSION/);
  assert.match(provision, /digest\(vcf_path\) == spec\["derived_sha256"\]/);
  assert.match(provision, /digest\(panel_path\) == panel_sha/);
  assert.match(provision, /secrets\.token_bytes\(32\)/);
  assert.match(provision, /n_samples.*== 500/);
  assert.match(provision, /sample_subpop/);
  assert.match(provision, /--category observed/);
  assert.match(task, /storage_mb = 32768/);
  assert.match(task, /environment_mode = "separate"/);
  assert.match(task, /network_mode = "no-network"/);
  assert.match(task, /artifacts = \["\/app\/submission"\]/);
  assert.doesNotMatch(task, /artifacts = .*\/logs/);
  assert.match(task, /\[verifier\.environment\]/);
  assert.match(task, /user = "root"/);
  assert.match(packagedTest, /REAL_VCF=\/opt\/hyperfocal\/pcabench-data\/chr22\.vcf/);
  assert.match(packagedTest, /mktemp -d \/tmp\/pcabench_grade\.XXXXXX/);
  assert.match(packagedTest, /trap 'rm -rf -- "\$\{WORK\}"' EXIT/);
  assert.match(packagedTest, /stat -c %d "\$\{REAL_VCF\}"/);
  assert.match(packagedTest, /len\(truth_paths\) == 22/);
  assert.match(packagedTest, /grade_observed\.vcf\.truth\.json/);
  assert.match(packagedTest, /--isolation verifier-container/);
  assert.match(packagedTest, /--representation-key-file "\$\{REPRESENTATION_KEY\}"/);
  assert.match(packagedTest, /--math-key-file "\$\{MATH_KEY\}"/);
  assert.match(packagedTest, /rm -f "\$\{REPRESENTATION_KEY\}"/);
  assert.match(packagedTest, /chmod 0755 \/tmp \/var\/tmp \/run\/lock \/dev\/shm \/dev\/mqueue/);
  assert.match(packagedTest, /os\.fchmod\(descriptor, 0o500\)/);
  assert.match(verifierImage, /FROM python:3\.12\.8-slim-bookworm/);
  assert.match(verifierImage, /a90c16c4ff2b3196476d506ae13cb3047fae8670163c7c932c4b0239aef3daf5/);
  assert.match(verifierImage, /b4023dc6ee2d62ee89c8d4d347db4d348e65518d66d346574cdae7a4bbd76858/);
  assert.match(verifierImage, /prepare_1kg\.py/);
  assert.match(verifierImage, /--representation-key-file \/tmp\/observed\.key/);
  assert.match(verifierImage, /--math-key-file \/opt\/hyperfocal\/pcabench-secrets\/release-v3\.math-key/);
  assert.match(verifierImage, /chmod 0400 \/opt\/hyperfocal\/pcabench-secrets\/release-v3\.math-key/);
  assert.match(verifierImage, /secrets\.token_bytes\(32\)/);
  assert.doesNotMatch(verifierImage, /bubblewrap/);
  assert.match(verifierImage, /chmod 0755 \/tmp/);
  assert.match(verifierImage, /chmod 0400 \/opt\/hyperfocal\/pcabench-data\/chr22\.vcf\.truth\.json/);
  assert.doesNotMatch(agentImage, /1000genomes|prepare_1kg|pcabench-data|grader_pkg/);
});

test("solver-visible prompts expose neither the observed cohort nor solution architecture", () => {
  for (const relative of ["environment/problems.yaml", "tasks/from_scratch_pca/instruction.md"]) {
    const visible = fs.readFileSync(path.join(root, relative), "utf8").toLowerCase();
    assert.doesNotMatch(visible, /1000 genomes|chr22|chromosome 22|observed cohort/);
    assert.doesNotMatch(visible, /\b(?:HG|NA)\d{4,}\b/);
    assert.doesNotMatch(visible, /bubblewrap|seccomp|execveat|system v ipc|chroot|python audit/);
    assert.doesNotMatch(
      visible,
      /sample[- ](?:by[- ]sample )?gram|byte[- ]offset|\bpread\b|\bssyrk\b|matrix[- ]free|fast reference/,
    );
    assert.doesNotMatch(
      visible,
      /speed factor|probe gate|category score|io_wide|sample_heavy|spectral_selection|very_high_rank/,
    );
    assert.doesNotMatch(
      visible,
      /confirm your environment|test it on vcfs you create yourself|simple working pca/,
    );
  }
});

test("production submission boundaries are explicit, state-clean, and fail closed", () => {
  // ZERO-CONFIG DEFAULT: the adapter selects verifier-container unless
  // PCABENCH_ISOLATION=bwrap is set, because harbor's verifier container lacks
  // CAP_SYS_ADMIN (bwrap cannot create namespaces there). Native EC2 rollouts opt
  // into the bwrap jail; nothing set == verifier-container.
  assert.match(adapter, /process\.env\.PCABENCH_ISOLATION === "bwrap"/);
  assert.match(adapter, /\?\s*"bwrap"\s*:\s*"verifier-container"/);
  assert.match(adapter, /--isolation \$\{ISOLATION\}/);
  for (const namespace of ["user", "net", "pid", "ipc", "uts"]) {
    assert.match(canonicalGrader, new RegExp(`"--unshare-${namespace}"`));
  }
  assert.match(canonicalGrader, /"--cap-drop", "ALL"/);
  assert.match(canonicalGrader, /"--die-with-parent"/);
  assert.match(canonicalGrader, /"--bind", str\(scratch\), "\/tmp"/);
  assert.match(canonicalGrader, /_BWRAP_WORK_ROOT = Path\("\/opt\/hyperfocal\/pcabench-work"\)/);
  assert.match(canonicalGrader, /"--bind", str\(shared_memory\), "\/dev\/shm"/);
  assert.match(canonicalGrader, /"--remount-ro", "\/"/);
  assert.match(resourceWatchdog, /_submission_fd_stats\(pid\)/);
  assert.match(resourceWatchdog, /os\.scandir\(f"\/proc\/\{pid\}\/map_files"\)/);
  assert.match(resourceWatchdog, /_retry_submission_fd_access\(/);
  assert.match(resourceWatchdog, /deadline = time\.monotonic\(\) \+ sum\(_FD_PERMISSION_RETRY_DELAYS\)/);
  assert.match(resourceWatchdog, /os\.setuid\(args\.uid\)/);
  assert.match(resourceWatchdog, /_aggregate_pss_bytes\(processes\)/);
  assert.match(canonicalGrader, /_decode_watchdog_report\(payload, proc\.pid\)/);
  assert.match(canonicalGrader, /submission resource monitor failed/);
  assert.doesNotMatch(canonicalGrader, /"--tmpfs", "\/tmp"/);
  assert.match(canonicalGrader, /_resource_watchdog_argv\(/);

  assert.match(task, /network_mode = "no-network"/);
  assert.match(verifierImage, /root=\/opt\/hyperfocal\/pcabench-sandbox-root/);
  assert.match(verifierImage, /install -d -m 0555 -o root -g root "\$root"/);
  assert.match(verifierImage, /cp -a \/usr "\$root\/usr"/);
  assert.match(verifierImage, /pcabench-guard\/submission_runner\.py/);
  assert.match(canonicalGrader, /_CHROOT,\s*str\(_CHROOT_ROOT\)/);
  assert.match(canonicalGrader,
    /"\/tests", "\/opt\/hyperfocal\/pcabench-data", "\/proc", "\/sys"/);
  assert.doesNotMatch(verifierImage, /\$root\/tests|\$root\/opt\/hyperfocal\/pcabench-data/);

  assert.match(submissionRunner, /def _install_no_exec_filter\(\)/);
  assert.match(submissionRunner, /def _install_no_new_executable_memory_filter\(\)/);
  assert.match(submissionRunner, /execve_number/);
  assert.match(submissionRunner, /execveat_number/);
  assert.match(submissionRunner, /unshare_number/);
  assert.match(submissionRunner, /setns_number/);
  assert.match(submissionRunner, /pr_set_dumpable = 4/);
  assert.match(submissionRunner, /mmap_number/);
  assert.match(submissionRunner, /mprotect_number/);
  assert.match(submissionRunner, /pkey_mprotect_number/);
  assert.match(submissionRunner, /shmat_number/);
  assert.match(submissionRunner, /seccomp_filter_flag_tsync = 1/);
  assert.match(submissionRunner, /sys\.addaudithook\(hook\)/);
  assert.match(submissionRunner, /event\.startswith\(\("ctypes\.", "gc\."\)\)/);
  assert.match(submissionRunner, /untrusted imported code path is disabled/);
  assert.match(submissionRunner, /sysconfig\.get_path\("platlib"\)/);
  assert.match(submissionRunner, /sysconfig\.get_path\("purelib"\)/);
  assert.match(submissionRunner, /sysconfig\.get_config_var\("DESTSHARED"\)/);

  assert.match(canonicalGrader, /Path\("\/proc\/sysvipc\/msg"\)/);
  assert.match(canonicalGrader, /Path\("\/proc\/sysvipc\/sem"\)/);
  assert.match(canonicalGrader, /Path\("\/proc\/sysvipc\/shm"\)/);
  assert.ok((canonicalGrader.match(/_terminate_submission_processes\(\)/g) ?? []).length >= 2);
  assert.ok((canonicalGrader.match(/_remove_submission_ipc\(\)/g) ?? []).length >= 3);
  assert.ok((canonicalGrader.match(/_clear_chroot_state\(\)/g) ?? []).length >= 3);
  assert.match(canonicalGrader,
    /for relative in \("tmp", "var\/tmp", "run\/lock", "dev\/shm", "dev\/mqueue"\)/);

  assert.match(submissionRunner,
    /sys\.path\.insert\(0, str\(submission_root\)\)/);
  assert.match(submissionRunner, /Linux ``fork``\s+workers remain usable/);
  assert.doesNotMatch(submissionRunner, /["']os\.fork["']/);
});
