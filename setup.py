from setuptools import find_packages, setup

NAME = "revnets"
version = "0.0.1"


def read(filename):
    try:
        with open(filename) as fp:
            content = fp.read().split("\n")
    except FileNotFoundError:
        content = []
    return content


setup(
    author="Judah Goldfeder & Quinten Roets",
    author_email="qdr2104@columbia.edu",
    description="",
    name=NAME,
    version=version,
    packages=find_packages(),
    setup_requires=read("setup_requirements.txt"),
    install_requires=read("requirements.txt"),
    entry_points={
        "console_scripts": [
            "revnets = revnets:main",
        ]
    },
)
