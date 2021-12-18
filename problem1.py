import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

n = 1024
p = 9

# 真实函数
alpha = 1.3
f = [
    lambda x: x / 3,
    lambda x: (x / 7) ** 2,
    lambda x: (x / 8) ** 3,
    lambda x: np.sin(x / 10 * np.pi),
    lambda x: np.log(np.abs(x + 13)) - 2.56494935746154,
    lambda x: (np.abs(x) / 8) ** 3.2 - (x / 8) ** 4,
    lambda x: np.abs(x / 5),
    lambda x: np.log(np.exp(x) + np.exp(x / 5) ** 2) / 5 - 0.138629436111989,
    lambda x: np.exp(x / 10) - 1
]

# 生成数据
points = np.arange(-7, 8, 1)
X = np.random.normal(loc=0, scale=6, size=(n, p))
y = alpha * np.ones((n, 1))
for j in range(p):
    y = y + f[j](X[:, [j]])
y = y + np.random.normal(loc=0, scale=1, size=(n, 1))


# 绘图模板
x_pred = np.linspace(-10, 10, 1024).reshape(-1, 1)
fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
for j in range(p):
    # axs[j // 3, j % 3].plot(x_pred, f_hat[j](x_pred)) # 估计函数
    axs[j // 3, j % 3].plot(x_pred, f[j](x_pred)) # 真实函数
plt.show()
