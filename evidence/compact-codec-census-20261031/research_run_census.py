"""Run the unchanged 64-step task with unscored saved-view observations."""

import os
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).absolute().parent
sys.path.insert(0, str(repo / "harness"))
from run_task import OFFLINE_ENV, make_config

out = repo / os.environ.get("SETWW_CENSUS_DIR", "runs_codec_census")
out.mkdir(exist_ok=True)
config = make_config(out, repo / "data/train.jsonl", 1)
env = dict(OFFLINE_ENV, PYTHONPATH=str(repo), SOUP_DB_PATH=str(out / "experiments.db"),
    TRITON_CACHE_DIR=str(out / "triton_cache"),
    TORCHINDUCTOR_CACHE_DIR=str(out / "inductor_cache"),
    SETWW_SPEED_PROFILE_REPORT=str(out / "lifecycle.json"),
    SETWW_CODEC_CENSUS_REPORT=str(out / "census.json"))
with (out / "diagnostic.log").open("w") as log:
    result = subprocess.run([sys.executable, str(repo / "research/research_codec_census.py"),
        "--solution", "--config", str(config), "--metrics", str(out / "metrics.json")],
        cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
print({"exit_code": result.returncode, "report": str(out / "census.json")}, flush=True)
raise SystemExit(result.returncode)
