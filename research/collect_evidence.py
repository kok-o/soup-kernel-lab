"""Collect selected outputs and source snapshots; omit weights/data/compiler caches."""

import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path


def main():
    from rich.console import Console

    root = Path(__file__).resolve().parents[1]
    destination = Path(sys.argv[1])
    names = {
        "result.json", "checks.json", "metrics.json", "profile.json", "diagnosis.json",
        "trainer_state.json", "adapter_config.json", "task.yaml", "rewind.jsonl",
        "solution-manifest.json", "solution-source.zip", "solution.zip",
        "head_microbench.json", "head_microbench.log", "integration_tests.log",
        "train.log", "run_task.log", "check_rules.log", "warmup.log",
    }
    roots = sorted(root.glob("runs_*")) + [root / "diagnostics",
                                          root / "evidence/head-fma-rejected"]
    paths = []
    for folder in roots:
        if not folder.is_dir():
            continue
        for parent, directories, filenames in os.walk(folder):
            directories[:] = [name for name in directories if name not in (
                "triton_cache", "inductor_cache", "__pycache__", ".cache"
            )]
            paths.extend(Path(parent) / name for name in filenames if (
                name in names or name.startswith("eval_") and name.endswith((".json", ".log"))
                or folder == root / "diagnostics" and name.endswith(".log")
            ))
    paths.extend(root.glob("*tests.log"))
    paths.extend(root.glob("*manifest.json"))
    paths.extend(root / name for name in ("SOURCE_MANIFEST.json", "PROVENANCE.json"))
    paths.extend((root / "research").glob("*.py"))
    paths = sorted(set(path for path in paths if path.is_file()))
    manifest = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths}
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("EVIDENCE_MANIFEST.json", json.dumps(manifest, indent=2))
    Console().print({"archive": str(destination), "files": len(paths),
                     "bytes": destination.stat().st_size,
                     "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})


if __name__ == "__main__":
    main()
