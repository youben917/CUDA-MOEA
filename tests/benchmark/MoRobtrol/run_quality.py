from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from launcher import launch
from plan import quality_jobs, smoke_jobs
if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    if smoke: sys.argv.remove("--smoke")
    jobs = smoke_jobs("smoke_d", checkpoints=True) if smoke else quality_jobs()
    raise SystemExit(launch(jobs, "Run MoRobtrol group D"))
