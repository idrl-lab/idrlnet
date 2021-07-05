# Installation

We recommend using conda to manage the environment.
Other methods may also work well such like using docker or virtual env.
## Anaconda

```bash
git clone https://git.idrl.site/pengwei/idrlnet
cd idrlnet
conda create -n idrlnet_dev python=3.8 -y
conda activate idrlnet_dev
pip install -r requirements.txt
pip install -e .
```
