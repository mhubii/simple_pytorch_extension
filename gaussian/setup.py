from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='gaussian_extension_cpp',
    ext_modules=[
        CppExtension('gaussian_extension_cpp', ['gaussian_extension.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
})
