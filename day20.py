import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False
X,y_true = make_blobs(n_samples=300,centers=3,cluster_std=0.60,random_state=42)
# make_blobs 是 sklearn 里专门用来生成“聚类测试数据”的神器（一坨一坨的）
# centers=4：生成 4 个中心点，也就是 4 个簇
# cluster_std=0.60：簇的标准差。数字越小，每个簇收缩得越紧（像一滴水滴）；数字越大，散得越开。0.60 表示每个簇的边界比较清晰，但略有重叠
# 一个 300 行 2 列的二维数组，代表 300 个点的坐标 (x, y)
# y_true：这 300 个点真实属于哪个簇（0, 1, 2, 3）
plt.scatter(X[:,0],X[:,1],s=50)
# X[:, 0]：所有点的 X 轴坐标,X[:, 1]：所有点的 Y 轴坐标
plt.title('原始数据')
plt.show()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.metrics import silhouette_score
inertia_values = []
# 创建一个空列表 inertia_values，用来装接下来算出来的成绩
k_range = range(1,11)
# range(1, 11) 表示生成 1 到 10 的整数。我们准备让模型分别尝试聚 1 类、2 类、3 类……一直到 10 类
for k in k_range:
    kmeans = KMeans(n_clusters=k,n_init=10,random_state=42)
    # n_clusters=k：强制让模型分出 k 个簇
    # n_init=10 意思是：“内部给我随机跑 10 次，选那个分得最好（误差最小）的结果
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)
# .append() 是 Python 列表的内置方法，意思是“追加/塞入”
#在 sklearn 里，带下划线的属性代表“模型训练（fit）之后才有的结果”
plt.plot(k_range,inertia_values,marker='o')
# inertia_values.append(kmeans.inertia_)
# marker='o' 会在每个点那里画个小圆圈，方便看
plt.title("肘部法寻找最佳 K 值")
plt.xlabel("K 值")
plt.ylabel("Inertia (误差平方和)")
plt.show()
silhouette_scores = []
k_range2 = range(2,11)
# 从2开始的数学原因：轮廓系数的计算需要一个核心参数：“该点到最近的其他簇的距离”
for k in k_range2:
    kmeans = KMeans(n_clusters=k,n_init=10,random_state=42)
    labels = kmeans.fit_predict(X_scaled)
# fit_predict：先执行 kmeans.fit(X_scaled) 训练模型，然后直接返回每个样本被分到的簇标签（labels）
    scores = silhouette_score(X_scaled,labels)
    silhouette_scores.append(scores)
plt.plot(k_range2,silhouette_scores,marker='o',color='red')
# 横轴是 K（2-10），纵轴是轮廓系数
plt.title("轮廓系数寻找最佳 K 值")
plt.xlabel("K 值")
plt.ylabel("轮廓系数")
plt.show()
kmeans_final = KMeans(n_clusters=3,n_init=10,random_state=42)
labels_2 = kmeans_final.fit_predict(X_scaled)
# labels2：是一个长度为 300 的一维数组，里面全是 0, 1, 2。比如 [0, 0, 2, 1, 0...]     
centers = kmeans_final.cluster_centers_
# cluster_centers_：这是 K-Means 训练完后学到的核心成果
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=labels_2,cmap='viridis',alpha=0.6)
# c=labels_2:把刚才算出来的标签数组传进去
plt.scatter(centers[:,0],centers[:,1],color='red',s=200,marker='X',edgecolors='black')
plt.title("K-Means 聚类结果 (K=3)")
plt.show()
