"""
Entry point: runs the full QFIS pipeline in order.
Step 1: Dataset preprocessing
Step 2: Build FAISS index
Step 3: Fine-tune (skipped if already done)
Step 4: Start API server
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def run(cmd, cwd=None):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or ROOT / "backend")
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="Skip fine-tuning")
    parser.add_argument("--only-api", action="store_true", help="Only start API server")
    args = parser.parse_args()

    py = sys.executable

    if not args.only_api:
        run([py, "training/dataset.py"])
        run([py, "rag/faiss_index.py"])
        if not args.skip_train:
            run([py, "training/finetune.py"])

    print("\n✓ Starting QFIS API server on http://localhost:8000")
    run([py, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
