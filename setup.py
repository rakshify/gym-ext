"""Setup script."""
import json
import setuptools

with open("README.md") as f:
    long_description = f.read()
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
with open("gym_ext/package_info.json") as f:
    pkg_info = json.load(f)

setuptools.setup(
    name=pkg_info["name"],
    include_package_data=True,
    version=pkg_info["version"],
    description=pkg_info["description"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=pkg_info["url"],
    packages=setuptools.find_packages(),
    classifiers=pkg_info["classifiers"],
    python_requires='>=3.7',
    install_requires=requirements,
    author=pkg_info["author"],
    author_email=pkg_info["author_email"],
)
