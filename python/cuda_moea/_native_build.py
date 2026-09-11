"""Linux native problem builder. Standard library only (also used by setup.py)."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _files(path: Path):
    if path.is_file():
        yield path
    else:
        for root, dirs, files in os.walk(path):
            dirs[:] = sorted(d for d in dirs if d not in {".git", "__pycache__"})
            for name in sorted(files):
                if not name.endswith(".pyc"):
                    yield Path(root) / name


def _inputs(source: Path, dependencies) -> dict:
    result = {}
    for path in [source, *map(lambda p: Path(p).resolve(), dependencies)]:
        if not path.exists():
            raise FileNotFoundError(path)
        for file in _files(path):
            result[str(file)] = _hash(file)
    return result


def sdk_directory() -> Path:
    return Path(__file__).resolve().parent / "sdk"


def cache_directory() -> Path:
    default = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "cuda_moea" / "native"
    return Path(os.environ.get("CUDA_MOEA_NATIVE_CACHE", default)).expanduser().resolve()


def _sdk(sdk: Path) -> dict:
    path = sdk / "build.json"
    if not path.is_file():
        raise RuntimeError("Native SDK missing; rebuild/install cuda_moea with native extension support")
    return json.loads(path.read_text())


def _architectures(value, sdk: dict) -> str:
    value = value or os.environ.get("CMAKE_CUDA_ARCHITECTURES") or sdk["architectures"]
    if not isinstance(value, str):
        value = ";".join(map(str, value))
    if not re.fullmatch(r"\d+(?:-(?:real|virtual))?(?:;\d+(?:-(?:real|virtual))?)*", value):
        raise ValueError("architectures must contain explicit CMake CUDA architectures, e.g. '89;90'")
    return value


def _tools() -> dict:
    result = {}
    for name, command in (("cmake", "cmake"), ("cxx", os.environ.get("CXX", "c++")),
                          ("nvcc", os.environ.get("CUDACXX", "nvcc"))):
        executable = shutil.which(command)
        if executable:
            try:
                version = subprocess.check_output([executable, "--version"], text=True, stderr=subprocess.STDOUT)
                result[name] = [str(Path(executable).resolve()), version.strip()]
            except (OSError, subprocess.CalledProcessError):
                pass
    return result


@contextlib.contextmanager
def _lock(cache: Path):
    if platform.system() != "Linux":
        raise RuntimeError("Native problem compilation currently supports Linux")
    import fcntl
    cache.mkdir(parents=True, exist_ok=True)
    # One lock also serializes cache cleanup with builds/readers. It lives outside
    # artifact directories so cleanup never replaces the inode being locked.
    with (cache / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _dependencies(build: Path) -> dict:
    """Capture compiler depfiles, including headers outside the problem tree."""
    result = {}
    for depfile in build.rglob("*.d"):
        try:
            body = depfile.read_text().replace("\\\n", " ").split(":", 1)[1]
            for token in shlex.split(body):
                path = Path(token)
                if not path.is_absolute():
                    path = build / path
                path = path.resolve()
                if path.is_file() and not path.is_relative_to(build):
                    result[str(path)] = _hash(path)
        except (OSError, ValueError, IndexError):
            # Explicit dependencies and source snapshots still apply; malformed
            # depfiles must not silently lead to an incompletely tracked build.
            raise RuntimeError(f"Cannot read compiler dependencies: {depfile}") from None
    return result


def _snapshot(source: Path) -> dict:
    return {str(p.relative_to(source)): _hash(p) for p in _files(source)}


def _bundled(source, sdk, identity, arch, debug, options):
    directory = sdk.parent / "native_problems" / source.name
    try:
        record = json.loads((directory / "artifact.json").read_text())
        request = record["request"]
        if (request["sdk"] == identity and request["architectures"] == arch and
                request["debug"] == bool(debug) and request["options"] == options and
                request["source_files"] == _snapshot(source) and
                _hash(directory / "problem.so") == record["library_hash"]):
            return directory / "problem.so"
    except (OSError, ValueError, KeyError):
        pass
    return None


def _valid(directory: Path, tools: dict) -> bool:
    try:
        record = json.loads((directory / "artifact.json").read_text())
        library = directory / "problem.so"
        if _hash(library) != record["library_hash"]:
            return False
        # A cached/prebuilt binary can be used on a runtime-only machine. When a
        # tool is present, a changed identity invalidates the compilation cache.
        if any(record["tools"].get(k) != v for k, v in tools.items()):
            return False
        for name, digest in record["dependencies"].items():
            path = Path(name)
            if path.exists() and _hash(path) != digest:
                return False
            if not path.exists() and "nvcc" in tools:
                return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def _run(command: list[str], log: Path, verbose: bool):
    with log.open("a") as stream:
        stream.write("$ " + shlex.join(command) + "\n")
        stream.flush()
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            assert process.stdout is not None
            for line in process.stdout:
                stream.write(line)
                if verbose:
                    print(line, end="", flush=True)
            status = process.wait()
    if status:
        tail = log.read_text()[-6000:]
        raise RuntimeError(f"Native problem build failed ({status}). Log: {log}\n{tail}")


def build_problem(source_dir, *, mode="auto", cache_dir=None, sdk_dir=None,
                  architectures=None, debug=False, force=False, verbose=False,
                  cmake_options=None, dependencies=(), jobs=None) -> Path:
    """Return an immutable cached .so, compiling only on a cache miss.

    dependencies explicitly fingerprints external libraries/data/build files.
    Runtime ProblemConfig values never enter this function or its cache key.
    """
    if mode not in {"auto", "never"}:
        raise ValueError("build must be 'auto' or 'never'")
    if force and mode == "never":
        raise ValueError("force cannot be combined with build='never'")
    source = Path(source_dir).expanduser().resolve()
    if not (source / "CMakeLists.txt").is_file():
        raise FileNotFoundError(f"Native problem requires {source / 'CMakeLists.txt'}")
    sdk = Path(sdk_dir).resolve() if sdk_dir else sdk_directory()
    identity = _sdk(sdk)
    cache = Path(cache_dir).expanduser().resolve() if cache_dir else cache_directory()
    if cache.is_relative_to(source):
        raise ValueError("cache_dir must be outside source_dir")
    arch = _architectures(architectures, identity)
    options = dict(cmake_options or {})
    reserved = {"CMAKE_BUILD_TYPE", "CMAKE_CUDA_ARCHITECTURES", "CudaMoeaNative_DIR",
                "CUDA_MOEA_PLUGIN_OUTPUT_DIR", "CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER",
                "CMAKE_MAKE_PROGRAM", "CMAKE_TOOLCHAIN_FILE", "CMAKE_PROJECT_INCLUDE"}
    for name in options:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) or name in reserved:
            raise ValueError(f"Reserved or invalid cmake_options key: {name}")
    if jobs is not None and (isinstance(jobs, bool) or not isinstance(jobs, int) or jobs <= 0):
        raise ValueError("jobs must be a positive integer")
    if not force and not dependencies:
        bundled = _bundled(source, sdk, identity, arch, debug, options)
        if bundled is not None:
            return bundled
    tools = _tools()
    request = {
        "format": 1, "inputs": _inputs(source, dependencies), "sdk": identity,
        "source_files": _snapshot(source),
        "architectures": arch, "debug": bool(debug), "options": options,
        "machine": platform.machine(),
        "environment": {k: os.environ.get(k, "") for k in (
            "CXX", "CUDACXX", "CXXFLAGS", "CUDAFLAGS", "LDFLAGS", "CMAKE_PREFIX_PATH",
            "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "CUDAHOSTCXX")},
    }
    key = _key(request)
    with _lock(cache):
        entries = cache / "artifacts" / key
        if not force and entries.exists():
            for directory in sorted(entries.iterdir(), key=lambda p: p.stat().st_mtime_ns, reverse=True):
                if _valid(directory, tools):
                    return directory / "problem.so"
        if mode == "never":
            raise FileNotFoundError("No compatible native problem cache; run 'python -m cuda_moea.native build SOURCE' or use build='auto'")
        if not {"cmake", "cxx", "nvcc"} <= tools.keys():
            raise RuntimeError("Building a native problem requires CMake, a C++ compiler and nvcc; set CXX/CUDACXX if needed")
        # Reuse the CMake work directory across source edits for incremental builds.
        work_key = _key({"source": str(source), "sdk": identity, "arch": arch,
                         "debug": debug, "options": options, "tools": tools,
                         "environment": request["environment"]})
        work = cache / "work" / work_key
        work.mkdir(parents=True, exist_ok=True)
        output = work / "output"
        log = work / "build.log"
        log.write_text("")
        cmake = tools["cmake"][0]
        configure = [cmake, "-S", str(source), "-B", str(work / "build"), "-G", "Unix Makefiles",
                     f"-DCudaMoeaNative_DIR={sdk / 'cmake'}",
                     f"-DCUDA_MOEA_PLUGIN_OUTPUT_DIR={output}",
                     f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
                     f"-DCMAKE_BUILD_TYPE={'Debug' if debug else 'Release'}",
                     f"-DCMAKE_CXX_COMPILER={tools['cxx'][0]}",
                     f"-DCMAKE_CUDA_COMPILER={tools['nvcc'][0]}"]
        configure += [f"-D{k}={v}" for k, v in sorted(options.items())]
        _run(configure, log, verbose)
        compile_command = [cmake, "--build", str(work / "build"), "--parallel", str(jobs or min(os.cpu_count() or 1, 8))]
        if force:
            compile_command.append("--clean-first")
        # Content changes with preserved/backdated timestamps must also rebuild.
        previous = work / "last_request.json"
        previous_deps = work / "last_dependencies.json"
        if previous.exists() and json.loads(previous.read_text()) != request and "--clean-first" not in compile_command:
            old = json.loads(previous.read_text()).get("inputs", {})
            changed = [Path(p) for p, h in request["inputs"].items() if old.get(p) != h]
            if any(p.stat().st_mtime_ns <= previous.stat().st_mtime_ns for p in changed):
                compile_command.append("--clean-first")
        if previous_deps.exists() and "--clean-first" not in compile_command:
            old_deps = json.loads(previous_deps.read_text())
            if any(Path(p).exists() and _hash(Path(p)) != digest and
                   Path(p).stat().st_mtime_ns <= previous_deps.stat().st_mtime_ns
                   for p, digest in old_deps.items()):
                compile_command.append("--clean-first")
        _run(compile_command, log, verbose)
        library = output / "problem.so"
        if not library.is_file():
            raise RuntimeError(f"Build did not produce {library}; use cuda_moea_add_problem()")
        if request["inputs"] != _inputs(source, dependencies):
            raise RuntimeError("Native problem sources changed during compilation; retry")
        record = {"request": request, "tools": tools,
                  "dependencies": _dependencies(work / "build"),
                  "library_hash": _hash(library)}
        entries.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=entries))
        try:
            shutil.copy2(library, staging / "problem.so")
            (staging / "artifact.json").write_text(json.dumps(record, indent=2, sort_keys=True))
            shutil.copy2(log, staging / "build.log")
            destination = entries / uuid.uuid4().hex
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        previous.write_text(json.dumps(request, sort_keys=True))
        previous_deps.write_text(json.dumps(record["dependencies"], sort_keys=True))
        return destination / "problem.so"


def clear_cache(cache_dir=None):
    cache = Path(cache_dir).resolve() if cache_dir else cache_directory()
    with _lock(cache):
        for name in ("artifacts", "work"):
            path = cache / name
            if path.exists():
                shutil.rmtree(path)


def export_problem(library, destination) -> Path:
    """Copy an immutable build into a distributable directory."""
    library = Path(library).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("problem.so", "artifact.json"):
        target = destination / name
        if target.exists():
            raise FileExistsError(f"Export destination already contains {target}; choose a new directory")
    shutil.copy2(library, destination / "problem.so")
    shutil.copy2(library.parent / "artifact.json", destination / "artifact.json")
    return destination / "problem.so"
