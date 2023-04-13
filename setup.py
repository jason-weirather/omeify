# setup.py
from setuptools import setup, find_packages

setup(
    name='omeify',
    version='0.1.0',
    author='Jason L Weirather',
    author_email='JasonL_Weirather@dfci.harvard.edu',
    license='MIT',
    description='A tool for converting and deidentifying image files into OME tiff format',
    packages=find_packages(),
    install_requires=[
        # Add your project dependencies here
        'xmltodict',
        'ipytree',
        'zarr'
    ],
    entry_points={
        'console_scripts': [
            'omeify=omeify.cli:main',
        ],
    },
)
