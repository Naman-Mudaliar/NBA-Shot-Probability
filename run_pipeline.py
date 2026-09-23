"""One-command runner: data -> features -> model -> dashboard payload."""
import subprocess
import sys

STEPS = ["generate_data.py", "features.py", "train_model.py", "build_dashboard_data.py"]

for step in STEPS:
    print(f"\n=== Running {step} ===")
    result = subprocess.run([sys.executable, step])
    if result.returncode != 0:
        sys.exit(result.returncode)

print("\nPipeline complete. Rebuild dashboard/dashboard.html by re-running the "
      "embed step in README.md (or open the existing dashboard.html directly).")
