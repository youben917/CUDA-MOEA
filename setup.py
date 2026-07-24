from __future__ import annotations

import os
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
            ["cmake", "--build", str(build), "--config", config, "-j"],
            env=build_env,
        )


setup(
    ext_modules=[CMakeExtension("cuda_moea._C")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
