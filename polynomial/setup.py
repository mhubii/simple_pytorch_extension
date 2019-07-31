from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='simple_extension_cpp',
    ext_modules=[
        CppExtension('simple_extension_cpp', ['simple_extension.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
})
