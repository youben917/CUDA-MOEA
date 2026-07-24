from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from plan import dimension_timing_jobs, smoke_jobs
from launcher import launch

if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    if smoke:
        sys.argv.remove("--smoke")
    jobs = smoke_jobs("smoke_c", save_objectives=False) if smoke else dimension_timing_jobs()
    raise SystemExit(launch(jobs, "Run DTLZ group C"))
