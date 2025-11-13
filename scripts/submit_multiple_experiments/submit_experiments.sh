#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 path/to/experiments.jsonl"
  exit 1
fi

EXPERIMENTS_FILE="$1"

if [ ! -f "$EXPERIMENTS_FILE" ]; then
  echo "File not found: $EXPERIMENTS_FILE"
  exit 1
fi

EXPERIMENTS_DIR="$(cd "$(dirname "$EXPERIMENTS_FILE")" && pwd)"
LOG_DIR="$EXPERIMENTS_DIR/logs"
mkdir -p "$LOG_DIR"

echo "Reading experiments from: $EXPERIMENTS_FILE"
echo "Logs will be written to: $LOG_DIR"
echo

python - << 'EOF' "$EXPERIMENTS_FILE" "$LOG_DIR"
import json
import os
import sys
import subprocess
from pathlib import Path
import re
from datetime import datetime

path = Path(sys.argv[1])
log_dir = Path(sys.argv[2])

lines = []
with path.open() as f:
    for i, raw in enumerate(f, 1):
        stripped = raw.strip()
        # Skip empty / whitespace-only lines and comments
        if not stripped or stripped.startswith("#"):
            continue
        lines.append((i, stripped))

if not lines:
    print("No experiments found in JSONL file.", file=sys.stderr)
    sys.exit(1)

procs = []

def sanitize_filename(name: str) -> str:
    name = name.strip().lower().replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    return name or "experiment"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for i, (lineno, line) in enumerate(lines, 1):
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSONL on line {lineno}: {e}", file=sys.stderr)
        sys.exit(1)

    name = data.get("experiment_name", f"experiment_{i}")
    cmd = data["bash"]
    gpu = str(data.get("gpu", "0"))

    safe_name = sanitize_filename(name)
    log_path = log_dir / f"{timestamp}_{i:02d}_{safe_name}.log"

    print(f"[Line {lineno}] Launching: {name} on GPU {gpu}")
    print(f"  Command: {cmd}")
    print(f"  Log: {log_path}")
    print()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    # Make Python logs unbuffered if using python
    env.setdefault("PYTHONUNBUFFERED", "1")

    log_file = open(log_path, "w")
    p = subprocess.Popen(
        cmd,
        shell=True,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    procs.append((name, p, log_file))

print("All experiments launched. Waiting for them to finish...")

for name, p, log_file in procs:
    rc = p.wait()
    log_file.close()
    print(f"Experiment '{name}' finished with return code {rc}")
EOF
