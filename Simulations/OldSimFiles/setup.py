#!/usr/bin/env python
# Thanks and credits to https://github.com/navdeep-G/setup.py for setup.py format
# To use: update the information (must update the version number), and use 
#         `$ python setup.py upload` command. Should already have twine.
import io
import os
import sys
# from shutil import rmtree

from setuptools import Command, find_packages, setup

setup(
    name = "fiber",
    version = '0.0.1',
    install_requires=["gym==0.21.0", 
                      "pyelastica==0.3.1.post1", 
                      "matplotlib==3.6.2", 
                      "pyglet==1.5.27", 
                      "vapory==0.1.2",
                      "pandas",
                      "cloudpickle==2.2.0",
                      "numpy==1.22.4",
                      "contourpy==1.0.6",
                      "cycler==0.11.0",
                      "fonttools==4.38.0",
                      "kiwisolver==1.4.4",
                      "packaging==22.0",
                      "pillow==9.3.0",
                      "pyparsing==3.0.9",
                      "python-dateutil==2.8.2",
                      "numba==0.55.2",
                      "scipy==1.9.3",
                      "tqdm==4.64.1",
                      "llvmlite==0.38.1",
                      "six==1.16.0",
                      "torch==1.13.1",
                      "stable_baselines3==1.6.2",
                      "torchvision==0.14.1"]
)

"""

from gym_softrobot.version import VERSION

# Package meta-data.
NAME = "gym-softrobot"
DESCRIPTION = "Soft-robotics control environment package for OpenAI Gym"
URL = "https://github.com/skim0119/gym-softrobot"
EMAIL = "skim449@illinois.edu"
AUTHOR = "skim0119"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = f"{VERSION}"

# What packages are required for this module to be executed?
REQUIRED = ["gym>=0.21.0", "pyelastica>=0.2.0", "matplotlib>=3.3.2", "pyglet>=1.4.0", "vapory"]

# What packages are optional?
EXTRAS = {}
    #"Optimization and Inverse Design": ["cma"]

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

class UploadCommand(Command):
    """"""Support setup.py upload.""""""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """"""Prints things in bold.""""""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system(f'git tag v{VERSION}')
        os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",  # This is important!
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*",
        "*/tests", "*/tests/*", "tests/*"]),
    #py_modules=['gym_softrobot'],
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Development Status :: 2 - Pre-Alpha",
        "Framework :: Robot Framework :: Library",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
    ],
    download_url=f"https://github.com/skim0119/gym-softrobot/archive/refs/tags/v{VERSION}.tar.gz",
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    # $ setup.py publish support
    cmdclass={
        'upload': UploadCommand,
    },
)
"""

