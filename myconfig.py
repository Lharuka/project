import torch

class MyConfig():
    """
    配置文件路径和训练相关参数
    """
    def __init__(self):
        self.data_path = r"data\dataset0.txt"
        self.results_path = r"results"
        self.train_ratio = 0.7
        self.label_pos = 0 # 标签所在列数
        self.batch_size = 2048
        self.num_epoch = 1000
        self.lr = 5e-5

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = None

if __name__ == "__main__":
    config = MyConfig()
    print(config.device)