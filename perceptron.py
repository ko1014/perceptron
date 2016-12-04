import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    # パーセプトロンの分類器

    # 学習率
    mu = float
    # トレーニング回数
    n_iterator = int
    # 重み
    w = [0, 0]
    # 切片
    b = float

    # construct
    def __init__(self, mu = 0.01, n_iterator = 10, w = [1, -1], b = 0):
        self.mu = mu
        self.n_iterator = n_iterator
        self.w = w
        self.b = b

    # 識別関数
    def predict_func(self, x):
        out = np.dot(self.w, x)
        if  out >= 0:
            res = 1
        else:
            res = -1
        return [res, out]

    # ヒンジ損失calc
    def hindhi_loss_func(self, x, y):
        # 負の場合、更新が必要
        [res, out] = self.predict_func(x)
        return max([0, y * out + self.b])

    # main
    def main(self, x, y):
        # 重みを更新する必要があった場合
        if self.hindhi_loss_func(x, y) < 0:
            self.w_update()
            self.w_update_func(x, y)
            self.b_update_func(x, y)

    # 重みをupdate
    def w_update_func(self, x, y):
        self.w = map(lambda i: i + i * self.mu * y * x, self.w)

    # 切片calc
    def b_update_func(self, x):
        self.b = b + numpy.linalg.norm(x)

    # 重みデータ取得
    def get_w(self):
        return self.w

item_num=100
loop = 1000
# 第1象限と第3象限のランダム座標
x1_1=np.ones(int(item_num/2))+10*np.random.random(int(item_num/2))
x1_2=np.ones(int(item_num/2))+10*np.random.random(int(item_num/2))
x2_1=-np.ones(int(item_num/2))-10*np.random.random(int(item_num/2))
x2_2=-np.ones(int(item_num/2))-10*np.random.random(int(item_num/2))
x1 = np.c_[x1_1, x1_2]
x2 = np.c_[x2_1, x2_2]
x = np.array(np.r_[x1, x2])
# 教師ラベルを1 or -1で振って1つのベクトルにまとめる
y1 = np.ones(int(item_num/2))
y2 = -1*np.ones(int(item_num/2))
y = np.array(np.r_[y1 ,y2])
perceptron = Perceptron()
for i in range(item_num):
    perceptron.main(x[i,:], y[i])
wvec = perceptron.get_w()

# 分離関数
x_range = range(-15, 16)
w_data = [np.dot(wvec, xi) for xi in x_range]
# x1[:,0] = x1のx1[0]だけを表示する,x1[:,1] = x1のx1[1]だけを表示する
plt.scatter(x1[:,0],x1[:,1],marker='o',color='g',s=100)
plt.scatter(x2[:,0],x2[:,1],marker='s',color='b',s=100)
plt.plot(x_range, w_data)
plt.show()
