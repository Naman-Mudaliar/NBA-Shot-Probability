"""
run_pipeline.py
---------------
  python run_pipeline.py --synthetic   # FAST LOOP: run after every edit (seconds, no API)
  python run_pipeline.py               # real data: uses data/shots_raw.parquet

Never fetches. Pull real data explicitly with `python fetch_data.py`, and only
once the synthetic loop is green.
"""
import argparse
import subprocess
import sys

import config

parser = argparse.ArgumentParser()
parser.add_argument("--synthetic", action="store_true", help="generate synthetic data instead of using real pulls")
args = parser.parse_args()
flag = ["--synthetic"] if args.synthetic else []

if args.synthetic:
    steps = [["generate_data.py"]]
else:
    if not config.REAL_SHOTS.exists():
        sys.exit(f"{config.REAL_SHOTS.relative_to(config.ROOT)} missing -- run `python fetch_data.py` first.")
    steps = []
steps += [["features.py", *flag], ["train_model.py", *flag]]
steps += [["-m", "pytest", "-q"]]

for step in steps:
    print(f"\n=== {' '.join(step)} ===", flush=True)
    result = subprocess.run([sys.executable, *step], cwd=config.ROOT)
    if result.returncode != 0:
        sys.exit(f"FAILED: {' '.join(step)}")

print("\nPipeline green.")
