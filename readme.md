### 项目说明
莫名其妙的深度学习回归任务。
### 环境配置
```shell
conda create -n myproject python=3.12
conda activate myproject
pip install -r requirements.txt
```
没有测试过，按需安装其他可能需要的库。
### 运行
训练：运行`train.ipynb`。

推理：运行`pred.ipynb`。

数据位于`data`目录，配置文件为`myconfig.py`，数据加载为`mydataloader.py`，模型文件为`mymodel.py`，训练结果保存在`result`目录。