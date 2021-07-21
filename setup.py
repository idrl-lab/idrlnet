import setuptools
import os
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")


def load_requirements(path_dir=here, comment_char="#"):
    with open(os.path.join(path_dir, "requirements.txt"), "r") as file:
        lines = [line.strip() for line in file.readlines()]
    requirements = []
    for line in lines:
        # filer all comments
        if comment_char in line:
            line = line[: line.index(comment_char)]
        if line:  # if requirement is not empty
            requirements.append(line)
    return requirements


setuptools.setup(
    name="idrlnet",  # Replace with your own username
    version="0.0.1",
    author="Intelligent Design & Robust Learning lab",
    author_email="weipeng@deepinfar.cn",
    description="IDRLnet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/idrl-lab/idrlnet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=load_requirements(),
)
