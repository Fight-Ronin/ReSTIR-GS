from __future__ import annotations

import os
import shutil

from torch.utils.cpp_extension import CUDA_HOME


print("CUDA_HOME", CUDA_HOME)
print("nvcc", shutil.which("nvcc"))
print("cl", shutil.which("cl"))
print("TORCH_EXTENSIONS_DIR", os.environ.get("TORCH_EXTENSIONS_DIR"))
