from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
	name='PyTorch-metric',
	description='A PyTorch module for the Wasserstein and Chamfer metrics', 
	ext_modules=[
		CUDAExtension(
			name='shape_measure', 
			sources=[
				'src/metrics.cpp',
				'src/emd.cu',
				'src/chamfer.cu',
			],
			include_dirs=['src'],
			libraries=["cusolver", "cublas"],
		),
	],
	cmdclass={'build_ext': BuildExtension}, 
)
