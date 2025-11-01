# Installation (Windows GPU)

### 1. Install conda

Link: https://conda.io/projects/conda/en/latest/user-guide/getting-started.html



### 2. Create environment

```shell
conda create -n ActiveCatal python=3.11
conda activate ActiveCatal
```



### 3. Install packages

GPU: GeForce GTX 1660 SUPER

GPU driver: [555.99](https://www.nvidia.cn/geforce/drivers/results/227381/)

[CUDA Toolkit 12.1 Downloads](https://developer.nvidia.com/cuda-12-1-0-download-archive)

[cuDNN 9.4.0 Downloads](https://developer.nvidia.com/cudnn-downloads)



```shell
conda install spyder
conda install mpltern
pip install pandas
pip install scikit-learn
pip install bayesian-optimization
conda install -c dglteam/label/cu121 dgl=2.0.0
conda install pytorch=2.3.0 torchvision=0.18.0 torchaudio=2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c dglteam/label/cu121 dgl=2.0.0
pip install tensorflow-cpu # (CPU only)
pip install agat
```





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
9.0.0 2.17.0 2.3.0 2.0.0+cu121 True
```











