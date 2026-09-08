import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import { test } from "node:test";
import ts from "typescript";

const loadDependency = createRequire(import.meta.url);

function load(filename) {
  const source = fs.readFileSync(new URL(filename, import.meta.url), "utf8");
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
      esModuleInterop: true,
    },
  });
  const exports = {};
  vm.runInNewContext(
    outputText,
    {
      exports,
      process,
      require: loadDependency,
    },
    { filename }
  );
  return exports;
}

const { listDanceClips } = load("dance-clips.ts");
const plain = (value) => JSON.parse(JSON.stringify(value));

function validClip(overrides = {}) {
  return {
    version: 1,
    name: "Fixture Dance",
    duration: 1.5,
    loop: true,
    keys: [
      { t: 0, joints: Array(14).fill(0), rootPitch: 0 },
      { t: 1.5, joints: Array(14).fill(0.1), rootPitch: -0.2 },
    ],
    ...overrides,
  };
}

function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "dance-clips-"));
  fs.mkdirSync(path.join(root, "dance-clip", "nested"), { recursive: true });
  fs.mkdirSync(path.join(root, "rlx", "artifacts", "run-a"), {
    recursive: true,
  });
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  return {
    root,
    write(relativePath, value) {
      const filePath = path.join(root, relativePath);
      fs.mkdirSync(path.dirname(filePath), { recursive: true });
      fs.writeFileSync(
        filePath,
        typeof value === "string" ? value : JSON.stringify(value)
      );
      return filePath;
    },
  };
}

test("lists authored clips and selected artifact references", async (t) => {
  const f = fixture(t);
  f.write(
    "dance-clip/nested/authored.clip.json",
    validClip({ name: "Authored clip" })
  );
  f.write(
    "dance-clip/unnamed.json",
    validClip({ name: "   ", duration: 2 })
  );
  f.write(
    "rlx/artifacts/run-a/reference.clip.json",
    validClip({ name: "Known reference", duration: 4 })
  );
  f.write("rlx/artifacts/run-a/evaluation.json", validClip());

  const clips = await listDanceClips({ workspaceRoot: f.root });

  assert.deepEqual(
    plain(clips),
    [
      {
        path: "dance-clip/nested/authored.clip.json",
        label: "Authored clip",
        durationSeconds: 1.5,
        frameCount: 2,
      },
      {
        path: "dance-clip/unnamed.json",
        label: "unnamed",
        durationSeconds: 2,
        frameCount: 2,
      },
      {
        path: "rlx/artifacts/run-a/reference.clip.json",
        label: "Known reference",
        durationSeconds: 4,
        frameCount: 2,
      },
    ]
  );
});

test("skips malformed clips without failing the catalog", async (t) => {
  const f = fixture(t);
  f.write("dance-clip/good.json", validClip());
  f.write("dance-clip/broken-json.json", "{");
  f.write("dance-clip/no-keys.json", validClip({ keys: [] }));
  f.write(
    "dance-clip/wrong-joints.json",
    validClip({ keys: [{ t: 0, joints: Array(13).fill(0) }] })
  );
  f.write(
    "dance-clip/out-of-order.json",
    validClip({
      keys: [
        { t: 0, joints: Array(14).fill(0) },
        { t: 1, joints: Array(14).fill(0) },
        { t: 0.5, joints: Array(14).fill(0) },
      ],
    })
  );
  f.write("dance-clip/too-long.json", validClip({ duration: 121 }));

  const clips = await listDanceClips({ workspaceRoot: f.root });

  assert.deepEqual(plain(clips.map((clip) => clip.path)), [
    "dance-clip/good.json",
  ]);
});

test("skips symlinks that escape the allowed clip roots", async (t) => {
  const f = fixture(t);
  const outside = path.join(os.tmpdir(), `outside-${path.basename(f.root)}.json`);
  fs.writeFileSync(outside, JSON.stringify(validClip({ name: "Outside" })));
  t.after(() => fs.rmSync(outside, { force: true }));
  fs.symlinkSync(outside, path.join(f.root, "dance-clip", "escaped.json"));

  const clips = await listDanceClips({ workspaceRoot: f.root });

  assert.deepEqual(plain(clips), []);
});

test("returns an empty catalog when optional roots do not exist", async (t) => {
  const f = fixture(t);
  fs.rmSync(path.join(f.root, "dance-clip"), { recursive: true, force: true });
  fs.rmSync(path.join(f.root, "rlx"), { recursive: true, force: true });

  assert.deepEqual(
    plain(await listDanceClips({ workspaceRoot: f.root })),
    []
  );
});
