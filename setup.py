from __future__ import annotations

import os
import runpy
import shutil
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext: Extension) -> None:
        import torch

        root = Path(__file__).resolve().parent
        build = Path(self.build_temp) / ext.name
        build.mkdir(parents=True, exist_ok=True)
        output = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        config = "Debug" if self.debug else "Release"
        args = [
            "cmake", "-S", str(root), "-B", str(build),
            f"-DCMAKE_BUILD_TYPE={config}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DPython_EXECUTABLE={os.sys.executable}",
            f"-DCUDA_MOEA_PYTHON_OUTPUT_DIR={output}",
        ]
        if "CMAKE_CUDA_ARCHITECTURES" in os.environ:
            args.append(
                "-DCMAKE_CUDA_ARCHITECTURES=" +
                os.environ["CMAKE_CUDA_ARCHITECTURES"]
            )
        build_env = os.environ.copy()
        if "TORCH_CUDA_ARCH_LIST" not in build_env:
            arch = build_env.get("CMAKE_CUDA_ARCHITECTURES", "89")
            build_env["TORCH_CUDA_ARCH_LIST"] = ";".join(
                value if "." in value else f"{value[:-1]}.{value[-1]}"
                for value in arch.split(";")
            )
        subprocess.check_call(args, env=build_env)
        subprocess.check_call(
            ["cmake", "--build", str(build), "--config", config,
             "--parallel", os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "8")],
            env=build_env,
        )
        # Optional problem sources are compiled by the same independent builder
        # used at runtime, against the SDK just generated for this wheel.
        builder = runpy.run_path(str(root / "python/cuda_moea/_native_build.py"))
        if (output / "native_problems").exists():
            shutil.rmtree(output / "native_problems")
        destinations = set()
        for source in filter(None, os.environ.get("CUDA_MOEA_NATIVE_PROBLEMS", "").split(os.pathsep)):
            source = Path(source).resolve()
            if source.name in destinations:
                raise ValueError("Bundled native problem directories must have unique basenames")
            destinations.add(source.name)
            library = builder["build_problem"](
                source, sdk_dir=output / "sdk", cache_dir=build / "native-cache",
                architectures=build_env.get("CMAKE_CUDA_ARCHITECTURES", "89"),
                debug=self.debug, verbose=True)
            destination = output / "native_problems" / source.name
            if destination.exists():
                shutil.rmtree(destination)
            builder["export_problem"](library, destination)

    def copy_extensions_to_source(self) -> None:
        super().copy_extensions_to_source()
        # setuptools otherwise copies only _C during editable/in-place builds.
        command = self.get_finalized_command("build_py")
        source = Path(command.get_package_dir("cuda_moea"))
        built = Path(self.build_lib) / "cuda_moea"
        for name in ("sdk", "lib", "native_problems"):
            if (built / name).exists() and (built / name).resolve() != (source / name).resolve():
                if (source / name).exists():
                    shutil.rmtree(source / name)
                shutil.copytree(built / name, source / name)
            elif name == "native_problems" and (source / name).exists() and not (built / name).exists():
                shutil.rmtree(source / name)


setup(
    ext_modules=[CMakeExtension("cuda_moea._C")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
