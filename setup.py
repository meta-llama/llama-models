# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import os

from setuptools import find_packages, setup


# Function to read a file
def read_file(file_path):
    with open(file_path) as file:
        return file.read()


# Function to read requirements from a requirements.txt file
def read_requirements(module_path):
    requirements_path = os.path.join(module_path, "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path) as req_file:
            return req_file.read().splitlines()
    return []


# Custom function to get package directories
def get_package_dirs(base_path):
    package_dirs = {
        "llama2": os.path.join(base_path, "llama2"),
        "llama3": os.path.join(base_path, "llama3"),
        "llama3_1": os.path.join(base_path, "llama3_1"),
    }
    return package_dirs


# Path to the directory containing the setup.py file
here = os.path.abspath(os.path.dirname(__file__))

# Get package directories dynamically
package_dirs = get_package_dirs(os.path.join(here, "models"))

# Collect requirements from all submodules
extras_require = {}
for package_name, package_path in package_dirs.items():
    extras_require[package_name] = read_requirements(package_path)


setup(
    name="llama_models",
    version="0.0.0.1",
    author="Meta Llama",
    author_email="rsm@meta.com",
    description="Llama model details",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/meta-llama/llama-models",
    # license=read_license(),
    # packages=find_packages(where="models"),
    package_dir={"llama_models": "models"},
    classifiers=[],
    python_requires=">=3.10",
    install_requires=[],
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        "llama2": ["LICENSE", "requirements.txt"],
        "llama3": ["LICENSE", "requirements.txt"],
        "llama3_1": ["LICENSE", "requirements.txt"],
    },
)
