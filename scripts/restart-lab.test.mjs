import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  chmod,
  cp,
  mkdtemp,
  mkdir,
  readFile,
  realpath,
  rm,
  writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sourceScript = path.join(repoRoot, "restart-lab.sh");

async function executable(file, contents) {
  await writeFile(file, contents);
  await chmod(file, 0o755);
}

async function fixture(options = {}) {
  const createdRoot = await mkdtemp(path.join(os.tmpdir(), "restart-lab-test-"));
  const root = await realpath(createdRoot);
  const bin = path.join(root, "mock-bin");
  const mock = path.join(root, "mock-state");
  const viewer = path.join(root, "duck-viewer");
  const lab = path.join(root, "microduck_local");
  const script = path.join(root, "restart-lab.sh");
  const log = path.join(mock, "commands.log");
  const killed = path.join(mock, "killed.log");

  await Promise.all([
    mkdir(bin, { recursive: true }),
    mkdir(mock, { recursive: true }),
    mkdir(path.join(viewer, "node_modules/next/dist/bin"), { recursive: true }),
    mkdir(path.join(viewer, "scripts"), { recursive: true }),
    mkdir(path.join(lab, ".venv/bin"), { recursive: true }),
    mkdir(path.join(root, "rlx"), { recursive: true }),
    ...(options.directVenv ? [mkdir(path.join(root, "rlx/.venv-microduck/bin"), { recursive: true })] : []),
  ]);
  await cp(sourceScript, script);
  await Promise.all([
    writeFile(log, ""),
    writeFile(killed, ""),
    writeFile(path.join(mock, "api-get-count"), "0\n"),
    writeFile(path.join(mock, "process-commands"), options.processCommands ?? ""),
    ...(options.processCommandsAfterPreflight === undefined
      ? []
      : [writeFile(
          path.join(mock, "process-commands-after-preflight"),
          options.processCommandsAfterPreflight,
        )]),
    writeFile(path.join(mock, "process-tree"), options.processTree ?? ""),
    writeFile(
      path.join(mock, "active.json"),
      JSON.stringify({ activeJob: options.activeJob ?? null }),
    ),
    writeFile(path.join(viewer, "node_modules/next/dist/bin/next"), ""),
    writeFile(path.join(viewer, "scripts/verify-lab-ready.mjs"), ""),
  ]);
  await executable(path.join(lab, ".venv/bin/duck-lab"), "#!/bin/sh\nexit 0\n");
  const python = path.join(root, "python3.12");
  await executable(python, "#!/bin/sh\nexit 0\n");
  if (options.directVenv) {
    await executable(
      path.join(root, "rlx/.venv-microduck/bin/python"),
      `#!/bin/bash
printf 'python-direct %s\\n' "$*" >> "$MOCK_LOG"
exit "\${MOCK_DIRECT_PYTHON_EXIT:-0}"
`,
    );
  }

  const bashEnv = path.join(mock, "bash-env");
  await writeFile(
    bashEnv,
    [
      "kill() {",
      '  printf "kill %s\\n" "$*" >> "$MOCK_LOG"',
      '  for argument in "$@"; do',
      '    case "$argument" in -*) ;; *) printf "%s\\n" "$argument" >> "$MOCK_KILLED" ;; esac',
      "  done",
      "}",
      "",
    ].join("\n"),
  );

  await executable(
    path.join(bin, "lsof"),
    `#!/bin/bash
printf 'lsof %s\\n' "$*" >> "$MOCK_LOG"
args=" $* "
if [[ "$args" == *" -d cwd "* ]]; then
  pid=""
  while (($#)); do
    if [[ "$1" == "-p" ]]; then pid="$2"; break; fi
    shift
  done
  cwd_file="$MOCK_STATE/process.$pid.cwd"
  [[ -f "$cwd_file" ]] && printf 'n%s\\n' "$(cat "$cwd_file")"
  exit 0
fi
if [[ "$args" =~ -iTCP:([0-9]+) ]]; then
  listener_file="$MOCK_STATE/listener.\${BASH_REMATCH[1]}"
  if [[ -f "$listener_file" ]]; then
    pid="$(cat "$listener_file")"
    if ! grep -qx "$pid" "$MOCK_KILLED" 2>/dev/null; then
      printf '%s\\n' "$pid"
      exit 0
    fi
  fi
  exit 1
fi
`,
  );

  await executable(
    path.join(bin, "ps"),
    `#!/bin/bash
printf 'ps %s\\n' "$*" >> "$MOCK_LOG"
if [[ "$*" == "-axo pid=,command=" ]]; then
  if [[ -f "$MOCK_STATE/process-commands-after-preflight" ]] &&
     grep -q '^uv ' "$MOCK_LOG"; then
    cat "$MOCK_STATE/process-commands-after-preflight"
  else
    cat "$MOCK_STATE/process-commands"
  fi
  exit 0
fi
if [[ "$*" == "-axo pid=,ppid=" ]]; then cat "$MOCK_STATE/process-tree"; exit 0; fi
pid=""
field=""
while (($#)); do
  case "$1" in
    -p) pid="$2"; shift 2 ;;
    -o) field="$2"; shift 2 ;;
    *) shift ;;
  esac
done
case "$field" in
  command=) [[ -f "$MOCK_STATE/process.$pid.command" ]] && cat "$MOCK_STATE/process.$pid.command" ;;
  ppid=) [[ -f "$MOCK_STATE/process.$pid.ppid" ]] && cat "$MOCK_STATE/process.$pid.ppid" || printf '1\\n' ;;
  stat=)
    grep -qx "$pid" "$MOCK_KILLED" 2>/dev/null && exit 1
    printf 'S\\n'
    ;;
esac
`,
  );

  await executable(
    path.join(bin, "curl"),
    `#!/bin/bash
printf 'curl %s\\n' "$*" >> "$MOCK_LOG"
[[ "\${MOCK_CURL_EXIT:-0}" == 0 ]] || exit "$MOCK_CURL_EXIT"
if [[ " $* " == *" -X POST "* ]]; then
  [[ "\${MOCK_FAIL_API_POST:-0}" == 0 ]] || exit 22
  printf '{"cancelled":true}\\n'
  exit 0
fi
case "\${*: -1}" in
  */joints) printf '{"joints":[0,1,2,3,4,5,6,7,8,9,10,11,12,13]}\\n' ;;
  */api/rlx)
    count="$(cat "$MOCK_STATE/api-get-count")"
    printf '%s\\n' "$((count + 1))" > "$MOCK_STATE/api-get-count"
    (( count < \${MOCK_FAIL_API_GETS:-0} )) && exit 22
    cat "$MOCK_STATE/active.json"
    ;;
  *) printf '{}\\n' ;;
esac
`,
  );

  await executable(
    path.join(bin, "node"),
    `#!/bin/bash
printf 'node %s\\n' "$*" >> "$MOCK_LOG"
if [[ "$1" == "-e" ]]; then exec "$REAL_NODE" "$@"; fi
printf 'node-argc %s\\n' "$#" >> "$MOCK_LOG"
for argument in "$@"; do printf 'node-arg <%s>\\n' "$argument" >> "$MOCK_LOG"; done
exit "\${MOCK_VERIFY_EXIT:-0}"
`,
  );
  await executable(
    path.join(bin, "uv"),
    `#!/bin/bash
printf 'uv %s\\n' "$*" >> "$MOCK_LOG"
exit "\${MOCK_UV_EXIT:-0}"
`,
  );
  await executable(
    path.join(bin, "nohup"),
    `#!/bin/bash
printf 'nohup %s\\n' "$*" >> "$MOCK_LOG"
exit 0
`,
  );
  await executable(
    path.join(bin, "npm"),
    `#!/bin/bash
printf 'npm %s\\n' "$*" >> "$MOCK_LOG"
exit 99
`,
  );

  async function addProcess(pid, { command, cwd = root, ppid = 1 } = {}) {
    await Promise.all([
      writeFile(path.join(mock, `process.${pid}.command`), `${command ?? ""}\n`),
      writeFile(path.join(mock, `process.${pid}.cwd`), `${cwd}\n`),
      writeFile(path.join(mock, `process.${pid}.ppid`), `${ppid}\n`),
    ]);
  }

  async function listener(port, pid, details) {
    await writeFile(path.join(mock, `listener.${port}`), `${pid}\n`);
    await addProcess(pid, details);
  }

  function run(args = [], env = {}) {
    const childEnv = {
      ...process.env,
      PATH: `${bin}:/usr/bin:/bin`,
      BASH_ENV: bashEnv,
      MOCK_STATE: mock,
      MOCK_LOG: log,
      MOCK_KILLED: killed,
      REAL_NODE: process.execPath,
      MICRODUCK_RESTART_TIMEOUT: "2",
      ...env,
    };
    if (options.explicitPython === false) {
      delete childEnv.MICRODUCK_STUDIO_PYTHON;
      delete childEnv.MICRODUCK_STUDIO_PYTHON_DIRECT;
    } else {
      childEnv.MICRODUCK_STUDIO_PYTHON = python;
    }
    return spawnSync("bash", [script, ...args], {
      cwd: root,
      encoding: "utf8",
      env: childEnv,
    });
  }

  return {
    root,
    viewer,
    lab,
    script,
    listener,
    process: addProcess,
    run,
    log: () => readFile(log, "utf8"),
    killed: () => readFile(killed, "utf8"),
    cleanup: () => rm(root, { recursive: true, force: true }),
  };
}

test("restart-lab.sh has valid Bash syntax and documents supported options", async () => {
  const syntax = spawnSync("bash", ["-n", sourceScript], { encoding: "utf8" });
  assert.equal(syntax.status, 0, syntax.stderr);

  const f = await fixture();
  try {
    const help = f.run(["--help"]);
    assert.equal(help.status, 0, help.stderr);
    assert.match(help.stdout, /--check/);
    assert.match(help.stdout, /--stop-jobs/);
    assert.match(help.stdout, /--readiness-only/);

    const invalid = f.run(["--fresh"]);
    assert.notEqual(invalid.status, 0);
    assert.match(invalid.stderr, /Unknown option: --fresh/);
    assert.doesNotMatch(await f.log(), /\bnpm\b|\buv\b|\bnohup\b/);
  } finally {
    await f.cleanup();
  }
});

test("foreign listener is refused before any stop, preflight, or start", async () => {
  const f = await fixture();
  try {
    await f.listener(63317, 91001, {
      command: "node next-server",
      cwd: path.join(f.root, "..", "another-workspace"),
    });
    const result = f.run();
    const log = await f.log();
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /belongs to another workspace/);
    assert.doesNotMatch(log, /^kill /m);
    assert.doesNotMatch(log, /^uv /m);
    assert.doesNotMatch(log, /^nohup /m);
  } finally {
    await f.cleanup();
  }
});

test("active API job is refused unless --stop-jobs is explicit", async () => {
  const f = await fixture({ activeJob: { kind: "train" } });
  try {
    await f.listener(63317, 91002, {
      command: "node next-server",
      cwd: f.viewer,
    });
    const result = f.run();
    const log = await f.log();
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Training\/evaluation\/rendering is active/);
    assert.match(log, /curl .*\/api\/rlx/);
    assert.doesNotMatch(log, /^kill /m);
    assert.doesNotMatch(log, /^uv /m);
    assert.doesNotMatch(log, /^nohup /m);
  } finally {
    await f.cleanup();
  }
});

test("unhealthy API is refused by default before preflight or shutdown", async () => {
  const f = await fixture();
  try {
    await f.listener(63317, 91007, {
      command: "node next-server",
      cwd: f.viewer,
    });
    const result = f.run([], { MOCK_FAIL_API_GETS: "1" });
    const log = await f.log();
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Existing RLX API is unresponsive/);
    assert.match(log, /curl .*\/api\/rlx/);
    assert.doesNotMatch(log, /curl .*\/joints/);
    assert.doesNotMatch(log, /^uv /m);
    assert.doesNotMatch(log, /^kill /m);
    assert.doesNotMatch(log, /^nohup /m);
  } finally {
    await f.cleanup();
  }
});

test("--stop-jobs tolerates only the pre-restart API GET failures", async () => {
  const f = await fixture();
  try {
    await f.listener(63317, 91008, {
      command: "node next-server",
      cwd: f.viewer,
    });
    await f.listener(8788, 91009, {
      command: "python duck-lab --port 8788",
      cwd: f.lab,
    });
    const result = f.run(["--stop-jobs"], { MOCK_FAIL_API_GETS: "2" });
    await new Promise((resolve) => setTimeout(resolve, 50));
    const log = await f.log();
    const killed = (await f.killed()).trim().split(/\s+/).filter(Boolean);
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
    assert.equal(
      (result.stderr.match(/API job state unavailable/g) ?? []).length,
      2,
    );
    assert.equal((log.match(/curl .*\/api\/rlx$/gm) ?? []).length, 3);
    assert.match(log, /curl -fsS --max-time 2 .*\/api\/rlx/);
    assert.doesNotMatch(log, /curl .* -X POST /);
    assert.deepEqual(killed.sort(), ["91008", "91009"]);
    assert.match(log, /^nohup .*duck-lab --port 8788/m);
    assert.match(log, /^nohup node .*\/next dev --hostname 127\.0\.0\.1 -p 63317/m);
  } finally {
    await f.cleanup();
  }
});

test("genuine local orphan PPO process appearing after preflight is refused", async () => {
  const f = await fixture({
    processCommandsAfterPreflight:
      "92001 /workspace/.venv/bin/python examples/ppo_microduck_studio.py train\n",
  });
  try {
    await f.process(92001, {
      command: "/workspace/.venv/bin/python examples/ppo_microduck_studio.py train",
      cwd: path.join(f.root, "rlx"),
      ppid: 1,
    });
    const result = f.run();
    const log = await f.log();
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Training\/evaluation\/rendering is active/);
    assert.match(log, /^uv /m);
    assert.equal((log.match(/^ps -axo pid=,command=$/gm) ?? []).length, 2);
    assert.match(log, /lsof -a -p 92001 -d cwd -Fn/);
    assert.doesNotMatch(log, /^kill /m);
    assert.doesNotMatch(log, /^nohup /m);
  } finally {
    await f.cleanup();
  }
});

test("monitor shell mentioning a Python PPO command is not treated as a job", async () => {
  const f = await fixture({
    processCommands:
      "92002 /bin/zsh -lc monitor python examples/ppo_microduck_studio.py train\n",
  });
  try {
    await f.process(92002, {
      command: "/bin/zsh -lc monitor python examples/ppo_microduck_studio.py train",
      cwd: f.root,
      ppid: 1,
    });
    const result = f.run();
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
    const log = await f.log();
    assert.equal((log.match(/^ps -axo pid=,command=$/gm) ?? []).length, 2);
    assert.doesNotMatch(log, /lsof -a -p 92002 -d cwd -Fn/);
    assert.doesNotMatch(log, /^kill .*92002$/m);
  } finally {
    await f.cleanup();
  }
});

test("orphan uv launcher is caught before its Python worker starts", async () => {
  const command = "uv run --isolated examples/ppo_microduck_studio.py train";
  const f = await fixture({ processCommands: `92003 ${command}\n` });
  try {
    await f.process(92003, { command, cwd: path.join(f.root, "rlx"), ppid: 1 });
    const result = f.run();
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Training\/evaluation\/rendering is active/);
    assert.doesNotMatch(await f.log(), /^kill |^nohup /m);
  } finally {
    await f.cleanup();
  }
});

test("preflight failure preserves existing services", async () => {
  const f = await fixture();
  try {
    await f.listener(63317, 91003, {
      command: "node next-server",
      cwd: f.viewer,
    });
    await f.listener(8788, 91004, {
      command: "python duck-lab --port 8788",
      cwd: f.lab,
    });
    const result = f.run([], { MOCK_UV_EXIT: "9" });
    const log = await f.log();
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /RLX runtime preflight failed; existing servers were not stopped/);
    assert.match(log, /^uv /m);
    assert.doesNotMatch(log, /^kill /m);
    assert.doesNotMatch(log, /^nohup /m);
  } finally {
    await f.cleanup();
  }
});

test("installed RLX venv Python is selected directly without invoking uv", async () => {
  const f = await fixture({ directVenv: true, explicitPython: false });
  try {
    const result = f.run();
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
    const log = await f.log();
    assert.match(log, /^python-direct .*\/scripts\/check-rlx-runtime\.py$/m);
    assert.doesNotMatch(log, /^uv /m);
  } finally {
    await f.cleanup();
  }
});

test("--stop-jobs cancels the API job and terminates scoped process trees", async () => {
  const f = await fixture({
    activeJob: { kind: "train" },
    processTree: [
      "91015 91005",
      "91025 91015",
      "91016 91006",
      "91998 91999",
      "",
    ].join("\n"),
  });
  try {
    await f.listener(63317, 91005, {
      command: "node next-server",
      cwd: f.viewer,
    });
    await f.listener(8788, 91006, {
      command: "python duck-lab --port 8788",
      cwd: f.lab,
    });
    await f.process(91999, {
      command: "node next-server",
      cwd: path.join(f.root, "..", "unrelated"),
    });

    const result = f.run(
      ["--stop-jobs", "--readiness-only"],
      { MOCK_FAIL_API_POST: "1" },
    );
    await new Promise((resolve) => setTimeout(resolve, 50));
    const log = await f.log();
    const killed = (await f.killed()).trim().split(/\s+/).filter(Boolean);
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
    assert.match(log, /curl .* -X POST .*\/api\/rlx/);
    assert.match(result.stderr, /API cancellation failed/);
    assert.deepEqual(killed.sort(), ["91005", "91006", "91015", "91016", "91025"]);
    assert.doesNotMatch(log, /kill .*91999/);
    assert.doesNotMatch(log, /kill .*91998/);
    assert.match(log, /^nohup .*duck-lab --port 8788/m);
    assert.match(log, /^nohup node .*\/next dev --hostname 127\.0\.0\.1 -p 63317/m);
    assert.match(log, /^node .*verify-lab-ready\.mjs .*--readiness-only/m);
    assert.doesNotMatch(log, /^npm /m);
    assert.doesNotMatch(log, /--fresh/);
  } finally {
    await f.cleanup();
  }
});

test("--check verifies existing services without stopping or starting them", async () => {
  const f = await fixture();
  try {
    const result = f.run(["--check"]);
    const log = await f.log();
    assert.equal(result.status, 0, result.stderr);
    assert.match(log, /curl .*\/joints/);
    assert.match(log, /^node .*verify-lab-ready\.mjs/m);
    assert.match(log, /^node-argc 5$/m);
    assert.doesNotMatch(log, /^node-arg <>$/m);
    assert.doesNotMatch(log, /^node-arg <--readiness-only>$/m);
    assert.doesNotMatch(log, /^kill /m);
    assert.doesNotMatch(log, /^uv /m);
    assert.doesNotMatch(log, /^nohup /m);
  } finally {
    await f.cleanup();
  }
});

test("verifier failures make --check fail nonzero", async () => {
  const f = await fixture();
  try {
    const result = f.run(["--check"], { MOCK_VERIFY_EXIT: "7" });
    assert.equal(result.status, 7);
    assert.match(await f.log(), /^node .*verify-lab-ready\.mjs/m);
  } finally {
    await f.cleanup();
  }
});
