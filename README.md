# IDRLnet

[![License](https://img.shields.io/github/license/analysiscenter/pydens.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.7/3.8/3.9-blue.svg)](https://python.org)
[![Documentation Status](https://readthedocs.org/projects/idrlnet/badge/?version=latest)](https://idrlnet.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/idrlnet.svg)](https://badge.fury.io/py/idrlnet)
[![DockerHub](https://img.shields.io/docker/pulls/idrl/idrlnet.svg)](https://hub.docker.com/r/idrl/idrlnet)
[![CodeFactor](https://www.codefactor.io/repository/github/idrl-lab/idrlnet/badge/master)](https://www.codefactor.io/repository/github/idrl-lab/idrlnet/overview/master)

**IDRLnet** is a machine learning library on top of [PyTorch](https://pytorch.org/). Use IDRLnet if you need a machine learning library that solves both forward and inverse differential equations via physics-informed neural networks (PINN). IDRLnet is a flexible framework inspired by [Nvidia Simnet](https://developer.nvidia.com/simnet>).

## Docs

- [Full docs](https://idrlnet.readthedocs.io/en/latest/)
- [Tutorial](https://idrlnet.readthedocs.io/en/latest/user/get_started/tutorial.html)
- Paper:
   - IDRLnet: A Physics-Informed Neural Network Library. [arXiv](https://arxiv.org/abs/2107.04320)

## Installation

Choose one of the following installation methods.

### PyPI

Simple installation from PyPI.

```bash
pip install -U idrlnet
```

Note: To avoid version conflicts, please use some tools to create a virtual environment first.

### Docker

Pull latest docker image from Dockerhub.

```bash
docker pull idrl/idrlnet:latest
docker run -it idrl/idrlnet:latest bash

```

Note: Available tags can be found in [Dockerhub](https://hub.docker.com/repository/docker/idrl/idrlnet).

### Anaconda

```bash
conda create -n idrlnet_dev python=3.8 -y
conda activate idrlnet_dev
pip install idrlnet
```

### From Source

```
git clone https://github.com/idrl-lab/idrlnet
cd idrlnet
pip install -e .
```


## Features

IDRLnet supports

-  complex domain geometries without mesh generation. Provided geometries include interval, triangle, rectangle, polygon, circle, sphere... Other geometries can be constructed using three boolean operations: union, difference, and intersection;
   ![Geometry](https://raw.githubusercontent.com/weipeng0098/picture/master/20210617081809.png)
-  sampling in the interior of the defined geometry or on the boundary with given conditions.
-  enables the user code to be structured. Data sources, operations, constraints are all represented by ``Node``. The graph will be automatically constructed via label symbols of each node. Getting rid of the explicit construction via explicit expressions, users model problems more naturally.
-  solving variational minimization problem;
   <img src="https://raw.githubusercontent.com/weipeng0098/picture/master/20210617082331.gif" alt="miniface" style="zoom:33%;" />
-  solving integral differential equation;
-  adaptive resampling;
-  recover unknown parameters of PDEs from noisy measurement data.

It is also easy to customize IDRLnet to meet new demands.

-  Main Dependencies

    -  [Matplotlib](https://matplotlib.org/)
    -  [NumPy](http://www.numpy.org/)
    -  [Sympy](https://https://www.sympy.org/)==1.5.1
    -  [pytorch](https://www.tensorflow.org/)>=1.7.0

## Contributing to IDRLnet

First off, thanks for taking the time to contribute!

-  **Reporting bugs.** To report a bug, simply open an issue in the GitHub "Issues" section.

-  **Suggesting enhancements.** To submit an enhancement suggestion for IDRLnet, including completely new features and minor improvements to existing functionality, let us know by opening an issue.
   
-  **Pull requests.** If you made improvements to IDRLnet, fixed a bug, or had a new example, feel free to send us a pull-request.
   
-  **Asking questions.** To get help on how to use IDRLnet or its functionalities, you can as well open an issue.

-  **Answering questions.** If you know the answer to any question in the "Issues", you are welcomed to answer.

## The Team

IDRLnet was originally developed by IDRL lab.

## Citation
Feel free to cite this library.

```bibtex
@article{peng2021idrlnet,
      title={IDRLnet: A Physics-Informed Neural Network Library}, 
      author={Wei Peng and Jun Zhang and Weien Zhou and Xiaoyu Zhao and Wen Yao and Xiaoqian Chen},
      year={2021},
      eprint={2107.04320},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
