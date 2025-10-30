"""Setup script for CUDA P2P extension."""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from torch.utils import cpp_extension

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
    name='kv_marketplace_p2p',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)

