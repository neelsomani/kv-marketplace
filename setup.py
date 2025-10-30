"""Setup script for geometry kv-marketplace package with CUDA extension."""

from pathlib import Path
from setuptools import setup, find_packages
from torch.utils import cpp_extension

# Read version from pyproject.toml or use default
readme_file = Path(__file__).parent / "README.md"

ext_modules = [
    cpp_extension.CUDAExtension(
        'kv_marketplace.transport.p2p_cuda',
        [
            'kv_marketplace/transport/p2p_cuda.cpp',
        ],
        extra_compile_args={
            'nvcc': ['-O3', '--use_fast_math'],
            'cxx': ['-O3'],
        }
    ),
]

setup(
    name='kv-marketplace',
    version='0.1.0',
    description='Cross-GPU KV Cache Marketplace for transformer inference',
    long_description=readme_file.read_text() if readme_file.exists() else '',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'xxhash>=3.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov',
        ],
    },
    zip_safe=False,
)

