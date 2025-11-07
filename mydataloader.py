import os
import numpy as np
import pandas as pd
import torch

class MyDataLoader:
    """
    读取txt或csv，返回按行打乱顺序的二维torch张量，
    
    数据类型为float32，与nn的参数默认类型一致
    """
    @staticmethod
    def load(path):
        ext = os.path.splitext(path)[1].lower()

        if ext == ".txt":
            arr = np.loadtxt(path)

        if ext == ".csv":
            arr = pd.read_csv(path).values

        rng = np.random.default_rng()
        rng.shuffle(arr)
        return torch.tensor(arr, dtype=torch.float32)

if __name__ == "__main__":
    arr = MyDataLoader.load(r"data\dataset0.txt")
    print("txt 内容:\n", arr)