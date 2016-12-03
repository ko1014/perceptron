import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    # パーセプトロンの分類器

    # 学習率
    mu: float
    # トレーニング回数
    n_iterator: int
    # 重み
    w
    # 切片
    b

    # construct
    def __init__(self, mu = 0.01, n_iterator = 10, w = [1, -1, 1], b = 0):
        self.mu = mu
        self.n_iterator = n_iterator
        self.w = w
        self.b = b

    # ヒンジ損失calc
    def hindhi_loss_func(self, w, x, y):
        # 負の場合、更新が必要
        return max([0, y * np.dot(w, x) + b])

    # main
    def main(self, x, y):
        # 重みを更新する必要があった場合
        if self.hindi_loss_func(x, y) < 0:
            self.w_update()
            self.w_update_func(x, y)
            self.b_update_func(x, y)

    # 重みをupdate
    def w_update_func(self, x, y):
        self.w = (0 until self.w.size).map(i => self.w(i) * self.mu * y * x(i))

    # 切片calc
    def b_update_func(self, x):
        self.b = b + numpy.linalg.norm(x)


