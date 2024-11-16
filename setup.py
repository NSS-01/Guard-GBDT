from setuptools import setup, find_packages
import os
path = os.path.dirname(os.path.abspath(__file__))
setup(
    name="NssMPClib",
    version="1.0",
    author="xxxx",
    author_email="",
    url="xxxx",
    license="MIT",
    packages=find_packages(where='.NssMPC'),
    include_package_data=True,
    install_requires=[f'torchcsprng @ file://localhost/{path}/csprng'],
)