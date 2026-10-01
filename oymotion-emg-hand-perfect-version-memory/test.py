# 分类模型
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取 (自动恢复形状 (1860, 500, 8))
train_data = np.load('./emg_data_labeled.npy', allow_pickle=True)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))

plt.plot(train_data)
plt.grid(True)
plt.show()