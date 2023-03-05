"""
setup.py: setup tool for package installation
---------------------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2022-01-11
* Version: 0.0.1

This is part of the MLProject (github.com/viebboy/mlproject)

License
-------
Apache License 2.0


"""

import setuptools
from mlproject.version import __version__


setuptools.setup(
    name="mlproject",
    version=__version__,
    author="Dat Tran",
    author_email="viebboy@gmail.com",
    description="Toolkit to build Machine Learning projects",
    long_description="Toolkit to build Machine Learning projects",
    long_description_content_type="text",
    license='LICENSE.txt',
    packages=setuptools.find_packages(),
    classifiers=['Operating System :: POSIX', ],
    entry_points={
        'console_scripts': [
            'mlproject = mlproject.cli:main',
        ]
    }
)
