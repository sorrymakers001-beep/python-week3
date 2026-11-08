import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False
housing = fetch_california_housing()
df = pd.DataFrame(housing.data,columns=housing.feature_names)
# housing.data：这是一个二维的 NumPy 数组，形状大概是 (20640, 8)，里面密密麻麻全是纯数字，代表 8 个特征
# pd.DataFrame(...)：召唤 Pandas 数据框，把上面这坨纯数字变成表格
# columns=housing.feature_names：就把 ['MedInc', 'HouseAge', ...] 这些字母标签贴到了对应的列上
print(df.head())
df['MedHouseVal'] = housing.target
df.info()
# 输出内容：行数、列数、每一列的名字、非空值数量（Non-Null Count）、数据类型（Dtype）、以及整个 DataFrame 占用的内存大小
df.describe()
# 输出内容：默认只针对数值型列，输出 8 个统计量：count (数量)、mean (均值)、std (标准差)、min (最小值)、25% / 50% / 75% (四分位数)、max (最大值)
print('缺失值统计：\n',df.isnull().sum())
# df.isnull()：把 DataFrame 变成一个全是 True（是缺失值）和 False（不是缺失值）的巨大矩阵
# .sum()：把每一列的 True 当成 1，False 当成 0，纵向加总。得出每一列到底缺了几个数据
plt.figure(figsize=(10,6))
sns.histplot(df['MedHouseVal'],bins=50,color='green',kde=True)
# kde=True：KDE 代表核密度估计（Kernel Density Estimation）。它会在直方图的柱子上方，画出一条平滑的曲线
plt.title("加州房价（MedHouseVal）分布直方图", fontsize=15)
plt.xlabel("房屋中位数价格（单位：十万美元）", fontsize=12)
plt.ylabel("房屋数量（频数）", fontsize=12)
plt.grid(axis='y', alpha=0.3) 
plt.show()
plt.figure(figsize=(12,8))
corr_matrix = df.corr()
# df.corr()：这是 Pandas 的绝招。它会自动挑出所有数值型的列，然后两两计算皮尔逊相关系数
# 相关系数的范围：永远在 -1 到 1 之间
sns.heatmap(corr_matrix,annot=True,fmt='.2f',cmap='coolwarm',linewidths=0.5)
# annot=True：Annotation，把具体的数字（如 0.68、-0.12）写在格子里
# cmap='coolwarm'：极其经典的配色方案（冷暖色）
plt.title('特征相关性热力图')
plt.show()

