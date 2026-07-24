from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from plan import quality_jobs, smoke_jobs
from launcher import launch

if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    if smoke:
        sys.argv.remove("--smoke")
    raise SystemExit(launch(smoke_jobs("smoke_a") if smoke else quality_jobs(), "Run DTLZ group A"))
