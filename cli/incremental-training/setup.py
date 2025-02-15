import setuptools
from incremental_training.version import __version__


setuptools.setup(
    name="incremental-training",
    version=__version__,
    author="Dat Tran",
    author_email="hello@dats.bio",
    description="Incremental model training tool built based on mlproject",
    long_description="Incremental model training tool built based on mlproject",
    long_description_content_type="text",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Operating System :: POSIX",
    ],
    entry_points={
        "console_scripts": [
            "incremental-training = incremental_training.cli:main",
        ]
    },
)
