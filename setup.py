import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="idrlnet",  # Replace with your own username
    version="0.0.1-rc1",
    author="Intelligent Design & Robust Learning lab",
    author_email="weipeng@deepinfar.cn",
    description="IDRLnet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/idrl-lab/idrlnet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
