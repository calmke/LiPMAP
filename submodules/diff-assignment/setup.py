from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_assignment",
    packages=['diff_assignment'],
    version='0.0.1',
    ext_modules=[
        CUDAExtension(
            name="diff_assignment._C",
            sources=[
                "ray_plane_assign.cu",
                "ext.cpp",
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
