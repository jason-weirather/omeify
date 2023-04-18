# setup.py
from setuptools import setup, find_packages

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='omeify',
    version='0.1.0',
    author='Jason L Weirather',
    author_email='jason.weirather@gmail.com',
    license='MIT',
    description='A tool for converting and deidentifying image files into OME tiff format',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        # Add your project dependencies here
        'xmltodict',
        'ipytree',
        'zarr',
        'tiff-inspector',
        'ome-schema'
    ],
    entry_points={
        'console_scripts': [
            'omeify=omeify.cli:main',
        ],
    },
    url="https://github.com/jason-weirather/omeify",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
