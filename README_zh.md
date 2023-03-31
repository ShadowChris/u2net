# 1. 运行结果
1. 下载并将checkpoint.pth放入checkpoints中：[OneDrive](https://1drv.ms/u/s!Am46hiIqzupmhq5V_Zt36ACGjmI6Xg?e=WPhq7y)

2. 打开get_foreground_demo_X.py（X表示版本，通常取最新的一版），查看并按需求输入单张图片
3. 运行get_foreground_demoX.py



# 2. 版本信息
Ver1: 输入图片，输出单张448 x 448的前景图


# 3. train.py：尝试训练
## 1. 报错信息
````
Traceback (most recent call last):
  File "E:\data\workspace\python\wanxiang-ai\u-2-net-portrait\train.py", line 14, in <module>
    from torch.utils.tensorboard import SummaryWriter
  File "E:\Anaconda\envs\u2net\lib\site-packages\torch\utils\tensorboard\__init__.py", line 4, in <module>
    LooseVersion = distutils.version.LooseVersion
AttributeError: module 'distutils' has no attribute 'version'
````
这个错误是由于在某些 Python 版本中，distutils 不再包含 version 模块。这可能与您当前的环境设置有关。

首先，请尝试升级 setuptools 包：
```
pip install --upgrade setuptools
```
然后，尝试再次运行您的代码。如果问题仍然存在，您可以尝试使用以下方法在 E:\Anaconda\envs\u2net\lib\site-packages\torch\utils\tensorboard\__init__.py 文件中进行更改：
找到以下行：
```
LooseVersion = distutils.version.LooseVersion
```

将其替换为：
````
from distutils.version import LooseVersion
````
保存文件并重新运行您的代码。这应该解决了问题。

## 2. 训练集路径 (待开发)
详见：conf/datasets.yaml