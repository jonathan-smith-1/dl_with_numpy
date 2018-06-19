from distutils.core import setup
from setuptools import find_packages

setup(
	name='dl_with_numpy',
	version='0.0.1',
	packages=find_packages('src'),
	package_dir={'': 'src'},
	url='https://github.com/jonathan-smith-1/dl_with_numpy',
	license='MIT License',
	author='Jonathan Smith',
	author_email='jhwsmith86@googlemail.com',
	description='Simple deep learning with numpy',
	install_requires=['numpy']
)
