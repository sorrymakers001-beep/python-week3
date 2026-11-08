import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Seaborn 是基于 Matplotlib 构建的高级数据可视化库
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
print(f"特征数：{X.shape[1]}")
print(f'样本数：{y.shape[0]}')  
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=1111,random_state=43)
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)
print("前8个样本的真实标签：", y_test[:8])
print("前8个样本的预测类别：", y_pred[:8])
print("前8个样本的概率分布：", y_proba[:8])
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
# 这行代码是调取 sklearn 的“阅卷工具包”，从不同维度给模型的成绩打分
acc = accuracy_score(y_test,y_pred)
# 模型总共猜对了百分之几
prec =precision_score(y_test,y_pred)
# 模型说你是“良性”（Positive=1）的人里，到底有多少是真的良性
rec = recall_score(y_test,y_pred,pos_label=0)
# 在所有真正的“恶性”病人里，模型成功找出了多少个
f1 = f1_score(y_test,y_pred)
# F1 分数是精确率和召回率的调和平均数
auc = roc_auc_score(y_test,y_proba[:,1])
# 模型区分“好”和“坏”的综合能力
print(f"准确率 (Accuracy): {acc:.3f}")
print(f"精确率 (Precision): {prec:.3f}")
print(f"召回率 (Recall): {rec:.3f}")
print(f"F1 分数: {f1:.3f}")
print(f"ROC-AUC: {auc:.3f}")
cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(8,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['mali(0)', 'tar(1)'], 
            yticklabels=['mali(0)', 'tar(1)'])
# cm：传入你之前算好的 2x2 混淆矩阵数组
# annot=True：Annotation（注释）。设为 True，它会把具体的数字直接写在方格子里
# fmt='d'：Format（格式）--显示整数
# cmap='Blues'：Color Map（颜色映射）。选择蓝色系
plt.xlabel('predictions')
plt.ylabel('trues')
plt.title('matrix')
plt.show()
fpr,tpr,threshold = roc_curve(y_test,y_proba[:,1])
# fpr (False Positive Rate - 假正率)：在真实为负类（0，恶性）的样本中，模型误判为正类（1，良性）的比例
# tpr (True Positive Rate - 真正率)：在真实为正类（1，良性）的样本中，模型正确预测为正类（1，良性）的比例
# thresholds：sklearn 自动生成的一系列阈值（比如从 0.9 降到 0.1）
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# fpr, tpr：这是你上一步算出来的横坐标（假正率，误诊率）和纵坐标（真正率，召回率
# lw=2：Line Width（线宽）
# label=f'ROC curve (AUC = {auc:.3f})'：极其巧妙的一步。
#   它使用了 Python 的 f-string 格式化字符串。
#   {auc:.3f} 表示把你前面算好的 AUC 变量放进来，并强制保留 3 位小数（比如把 0.98 显示为 0.980）。
#   这个 label 是给谁看的？是给后面的 plt.legend()（图例）看的。这样图表的右下角（或左下角）就会自动生成一个带有橙色线段和 "ROC curve (AUC = 0.98...)" 的小框框
# [0, 1], [0, 1]：传入的 x 和 y 坐标。它连接了点 (0, 0)（左下角）和点 (1, 1)（右上角）
plt.xlabel('Ture rate')
plt.ylabel('False rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
from sklearn.model_selection import GridSearchCV
# 定义要测试的参数字典
param_grid = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}
# 自动做交叉验证并寻找最佳参数
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print(f"最好的 C 是: {grid.best_params_['C']}")