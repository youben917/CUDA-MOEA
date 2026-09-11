"""Build, inspect and use C++/CUDA implementations of IProblemEvaluator."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from ._native_build import build_problem, clear_cache, export_problem, sdk_directory


def benchmark(problem, variables, *, repeats=100, warmup=10):
    """Time full batch evaluation (including adapters), excluding setup/build.

    CUDA events measure the evaluation-stream timeline including launch gaps;
    wall time additionally includes host scheduling and the final wait.
    """
    from . import _C
    return _C.benchmark_problem(problem._spec(), variables, repeats, warmup)


class NativeProblem:
    def __init__(self, source_dir=None, *, name: str, dimension: int,
                 objectives: int, lower_bounds=0.0, upper_bounds=1.0,
                 parameters=None, library=None, build="auto", **build_options):
        if (source_dir is None) == (library is None):
            raise ValueError("Provide exactly one of source_dir or library")
        if build not in {"auto", "never"}:
            raise ValueError("build must be 'auto' or 'never'")
        for key, value in (("dimension", dimension), ("objectives", objectives)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a nonempty string")
        lower = self._bounds(lower_bounds, dimension)
        upper = self._bounds(upper_bounds, dimension)
        if any(a >= b for a, b in zip(lower, upper)):
            raise ValueError("Each lower bound must be less than its upper bound")
        if library is not None and build_options:
            raise ValueError("Build options are only applicable to source_dir")
        if library is None:
            library = build_problem(source_dir, mode=build, **build_options)
        self.library = Path(library).expanduser().resolve()
        if not self.library.is_file():
            raise FileNotFoundError(self.library)
        from . import _C
        self.schema = _C.native_problem_schema(str(self.library))
        if self.schema["name"] != name:
            raise ValueError(f"Library registers {self.schema['name']!r}, not {name!r}")
        self._config = {"type": "NativeProblem", "library": str(self.library),
                        "name": name, "dimension": dimension, "objectives": objectives,
                        "lower_bounds": lower, "upper_bounds": upper,
                        "parameters": copy.deepcopy(dict(parameters or {}))}
        self._validate_parameters()

    @staticmethod
    def _bounds(value, dimension):
        values = [float(value)] * dimension if isinstance(value, (int, float)) else list(map(float, value))
        if len(values) != dimension or not all(math.isfinite(x) and abs(x) <= 3.4028234663852886e38 for x in values):
            raise ValueError("Bounds must be finite float32 values with length dimension")
        # Native ProblemInfo uses float32. Validate after conversion as well.
        import struct
        return [struct.unpack("f", struct.pack("f", x))[0] for x in values]

    def _validate_parameters(self):
        validators = {
            "int": lambda v: type(v) is int and -(2**63) <= v < 2**63,
            "float": lambda v: type(v) in (int, float) and math.isfinite(v),
            "bool": lambda v: type(v) is bool,
            "str": lambda v: isinstance(v, str),
            "float_list": lambda v: isinstance(v, (list, tuple)) and all(
                type(x) in (int, float) and math.isfinite(x) for x in v),
        }
        for key, value in self._config["parameters"].items():
            if key not in self.schema["parameters"]:
                raise ValueError(f"Unknown native problem parameter: {key}")
            kind = self.schema["parameters"][key]["type"]
            if not validators[kind](value):
                raise ValueError(f"Native parameter {key!r} must be {kind}")

    def _spec(self):
        return copy.deepcopy(self._config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Compile or reuse a native problem")
    build.add_argument("source_dir")
    build.add_argument("--cache-dir")
    build.add_argument("--architectures")
    build.add_argument("--debug", action="store_true")
    build.add_argument("--force", action="store_true")
    build.add_argument("--verbose", action="store_true")
    build.add_argument("--dependency", action="append", default=[])
    build.add_argument("--jobs", type=int)
    build.add_argument("--output", help="Export to a new directory for distribution")
    inspect = sub.add_parser("inspect", help="Show registered name and parameter schema")
    inspect.add_argument("library")
    clear = sub.add_parser("clean", help="Remove cached artifacts/workspaces; stop users first")
    clear.add_argument("--cache-dir")
    sub.add_parser("sdk", help="Print installed SDK directory")
    args = parser.parse_args(argv)
    if args.command == "build":
        path = build_problem(args.source_dir, cache_dir=args.cache_dir,
                             architectures=args.architectures, debug=args.debug,
                             force=args.force, verbose=args.verbose,
                             dependencies=args.dependency, jobs=args.jobs)
        if args.output:
            path = export_problem(path, args.output)
        print(path)
    elif args.command == "inspect":
        from . import _C
        print(json.dumps(_C.native_problem_schema(str(Path(args.library).resolve())), indent=2))
    elif args.command == "clean":
        clear_cache(args.cache_dir)
    else:
        print(sdk_directory())


if __name__ == "__main__":
    main()
