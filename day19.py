import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
tree_unlimited = DecisionTreeClassifier(random_state=42)
tree_unlimited.fit(X_train,y_train)
train_acc = tree_unlimited.score(X_train,y_train)
test_acc = tree_unlimited.score(X_test,y_test)
print(f'练:{train_acc},测：{test_acc}')
plt.figure(figsize=(25, 8))
plot_tree(tree_unlimited,max_depth=3,filled=True,feature_names=cancer.feature_names,
          class_names=['恶性','良性']
)
plt.show()
# max_depth=2 强制要求：只给我画出树的最顶层 3 层（根节点 1 层 + 子节点 2 层）
# filled=True：给节点填涂颜色。颜色越深，代表这个节点里的样本越“纯”
# feature_names：把底层的 X[0], X[1] 替换成医学名词
tree_limited = DecisionTreeClassifier(max_depth=3,criterion='entropy',random_state=42)
tree_limited.fit(X_train,y_train)
train_acc1 = tree_limited.score(X_train,y_train)
test_acc1 = tree_limited.score(X_test,y_test)
print(f'有限制版，训：{train_acc1}，测{test_acc1}')
plt.figure(figsize=(25, 8))
plot_tree(tree_limited,max_depth=3,filled=True,feature_names=cancer.feature_names,
)
plt.show()
from sklearn.ensemble import RandomForestClassifier
Rforest = RandomForestClassifier(max_depth=5,random_state=42)
Rforest.fit(X_train,y_train)    
forest_acc_train = Rforest.score(X_train,y_train)
forest_acc_test = Rforest.score(X_test,y_test)
print(f"随机森林，练：{forest_acc_train}，测：{forest_acc_test}")
import seaborn as sns
importances = Rforest.feature_importances_
feature_names = cancer.feature_names
df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# importances：随机森林算出来的 30 个重要度分数（总和为 1.0）
# pd.DataFrame 将它们两两配对，变成一张有两列（特征名、重要度）的表格
df_importance = df_importance.sort_values(by='Importance', ascending=False).head(10)
# head(10)：只保留前 10 行
# sort_values(by='Importance', ascending=False)：按重要度这一列，从大到小降序排列（False 表示不升序）
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=df_importance, palette='viridis')
# data=df_importance：直接把刚才排好序的 DataFrame 传进去，Seaborn 会自动去里面找
# palette='viridis'：一套极具科技感的渐变色（从深紫到明黄）
plt.title('Top 10 Most Important Features')
plt.show()
