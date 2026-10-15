from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ddpm_cuda",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="ddpm_cuda",
            sources=["csrc/ddpm.cpp", "csrc/ddpm_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
