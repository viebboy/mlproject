import setuptools
from ml.version import __version__


setuptools.setup(
    name="model-training",
    version=__version__,
    author="Dat Tran",
    author_email="hello@dats.bio",
    description="Model training tool built based on mlproject",
    long_description="Model training tool built based on mlproject",
    long_description_content_type="text",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Operating System :: POSIX",
    ],
    entry_points={
        "console_scripts": [
            "model-training = model-training.cli:main",
        ]
    },
)
