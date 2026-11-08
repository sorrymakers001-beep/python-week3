import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error
housing = fetch_california_housing()
X = housing.data
y = housing.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
models = {
    "线性回归":LinearRegression(),
    "Ridge":Ridge(alpha=1.0),
    "Lasso":Lasso(alpha=0.1)
}
for name,model in models.items() :
    model.fit(X_train_scaled,y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_pred,y_test))
    print(f"{name}的成绩：{rmse:.4f}")
lasso_model = models['Lasso']
lasso_model.fit(X_train,y_train)
zero = np.sum(lasso_model.coef_ == 0)
print(f'清零数量：{zero}')
for a in [0.01, 0.1, 1.0, 10.0]:
    lasso = Lasso(alpha=a)
    lasso.fit(X_train_scaled, y_train)
    y_pred = lasso.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
lasso_model = models['Lasso']
zeros = np.sum(lasso.coef_ == 0)
print(f" 清零特征数: {zeros}")
from sklearn.model_selection import KFold,StratifiedKFold,cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
lin_reg = LinearRegression()
# 召唤一个普通的线性回归模型。注意：此时模型里面是空的，没有经过任何训练
scores = cross_val_score(lin_reg,X_train_scaled,y_train,cv=5,scoring='neg_mean_squared_error')
# cv=5：把 X_train_scaled（假设有 1000 条数据）平均分成 5 份。每次拿其中 4 份做训练，剩下 1 份做测试。这个过程循环 5 次
# scoring='neg_mean_squared_error'：告诉函数“我要算均方误差（MSE）”
print(f"5次得分 (MSE): {-scores}") 
print(f"平均 MSE: {-scores.mean():.4f}")
# 后台把 5 次测试都跑完了，所以 scores 会返回一个包含 5 个数字的数组
# 必须手动加一个负号（-），把它转回人类能看懂的正数     

# StratifiedKFold 的魔法：它的全称叫“分层抽样”，
# 它在切分时，会保证每一折（包括训练集和验证集）里的良恶性比例，都和原始数据的比例（9:1）一模一样
cancer = load_breast_cancer()
X_c = cancer.data
y_c =cancer.target
X_c_train,X_c_test,y_c_train,y_c_test = train_test_split(X_c,y_c,test_size=0.2,random_state=42)
scaler_c = StandardScaler()
X_c_train_scaled = scaler_c.fit_transform(X_c_train)
X_c_test_scaled = scaler_c.transform(X_c_test)
log_reg = LogisticRegression(max_iter=1000)
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
# _splits=5：切 5 折
# shuffle=True：极其关键！ 也就是在切分前，先把原始数据打乱
scores_acc = cross_val_score(log_reg,X_c_train_scaled,y_c_train,cv= skf,scoring='accuracy')
# shuffle=True：极其关键！ 也就是在切分前，先把原始数据打
# cv=skf：把刚刚建好的分层切割机传进去
print(f"5次得分 (Accuracy): {scores_acc}")
print(f"平均 Accuracy: {scores_acc.mean():.4f}")
log_reg.fit(X_c_train_scaled,y_c_train)
final_score = log_reg.score(X_c_test_scaled,y_c_test)
print(f"最终测试集准确率: {final_score:.4f}")