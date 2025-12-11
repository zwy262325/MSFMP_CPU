import numpy as np
data = np.fromfile('C:\\Users\\10592\Desktop\BasicTS-master\datasets\METR-LA\data.dat', dtype=np.float32)
print(data)
print(data.shape)
# 我只想打印这个文件的前9个数据
print(data[:624])