"""scripts/run_monte_seeds.py

Usage:
  python scripts/run_monte_seeds.py 2025            # single seed
  python scripts/run_monte_seeds.py 2025 42         # multiple seeds

This wrapper loads config.yaml, injects the given seeds into cfg["montecarlo"]["seeds"],
limits threading for deterministic CPU usage and runs pipeline.montecarlo.run(cfg).
"""
import os
import sys
from pathlib import Path

# limit internal BLAS/OMP threads to avoid oversubscription when running multiple processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# limit PyTorch threads
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

import yaml

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_monte_seeds.py <seed1> [seed2 ...]")
        sys.exit(1)
    seeds = [int(s) for s in sys.argv[1:]]

    cfg_path = Path("config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError("config.yaml not found in current directory")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("montecarlo", {})["seeds"] = seeds

    print(f"Launching montecarlo for seeds: {seeds}")

    # Ensure project root is on sys.path so `pipeline` package can be imported
    project_root = Path(__file__).resolve().parents[1]
    import sys
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pipeline import montecarlo
    montecarlo.run(cfg)
