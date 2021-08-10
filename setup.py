# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os.path
from setuptools import setup, find_packages

readme = ""
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, "README.rst")
if os.path.exists(readme_path):
    with open(readme_path, "rb") as stream:
        readme = stream.read().decode("utf8")

setup(
    long_description=readme,
    name="sparsepyramids",
    version="0.0.2",
    description="Various sparsity and pyramid functions for neural networks.",
    python_requires="==3.*,>=3.6.0",
    project_urls={"repository": "https://simleak.com/simleek/sparse_pyramids"},
    author="SimLeek",
    author_email="simleak@simleak.com",
    license="MIT",
    packages=find_packages(),
    package_dir={"": "."},
    package_data={},
    install_requires=["numpy>=1.14.5", "torch>=1.9.0"],
    extras_require={
        "dev": [
            "black==18.*,>=18.3.0.a0",
        ],
    },
)
