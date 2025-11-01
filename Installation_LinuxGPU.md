# Installation (Windows GPU)

### 1. Install conda

Link: https://conda.io/projects/conda/en/latest/user-guide/getting-started.html



### 2. Create environment

```shell
conda create -n ActiveCatal python=3.11
conda activate ActiveCatal
module load compiler/cuda/12.4 # load cuda
```



### 3. Install packages

```shell
conda install spyder
pip install mpltern
pip install pandas
pip install bayesian-optimization
pip install scikit-learn
conda install -c dglteam/label/th24_cu124 dgl
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install tensorflow-cpu # (CPU only)
pip install agat
```

You can also download `dgl` wheel from: https://data.dgl.ai/wheels/cu121/repo.html



### 4. Test the environment

```python
import agat
import tensorflow as tf
import torch
import dgl
print(agat.__version__, tf.__version__, torch.__version__, dgl.__version__, torch.cuda.is_available())
```

Ouput:

```shell

```











