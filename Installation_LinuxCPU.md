# Installation (Linux CPU only)

### 1. Install conda

Link: https://conda.io/projects/conda/en/latest/user-guide/getting-started.html



### 2. Create environment

```shell
conda create -n ActiveCatal python=3.11
conda activate ActiveCatal
```



### 3. Install packages

```shell
conda install spyder
conda install mpltern
pip install pandas
pip install bayesian-optimization
pip install scikit-learn
# pip install pydantic
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/repo.html # CPU version
# conda install pytorch==2.3 torchvision torchaudio cpuonly -c pytorch # CPU version
# pip install torchdata
pip install tensorflow-cpu
pip install agat==8.0.5
```



### 4. Test the environment

```python
import agat
import tensorflow as tf
import torch
import dgl
print(agat.__version__, tf.__version__, torch.__version__, dgl.__version__)
```

Ouput:

```shell
9.0.0 2.17.0 2.4.0+cu121 2.4.0
```











