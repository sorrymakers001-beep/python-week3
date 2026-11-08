import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
house = fetch_california_housing()
X = house.data
y = house.target
df = pd.DataFrame(X,columns=house.feature_names)
df[(df['AveRooms'] > 5) & (df['HouseAge'] < 20)]
df['house price'] = y
print(df.head())
print(df.shape)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=91)
scaler = StandardScaler()
# StandardScaler()作用：把不同量级、不同单位的数据，强行拉到同一个数量级上
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
# 套用公式,对 X_train 里的每一个数字进行计算，并将结果输出给新的变量 X_train_scaled
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse) # RMSE 就是 MSE 开根号
mae = mean_absolute_error(y_test,y_pred)
r2 =r2_score(y_test,y_pred)
print(f"均方误差 (MSE): {mse:.3f}")
print(f"均方根误差 (RMSE): {rmse:.3f}")
print(f"平均绝对误差 (MAE): {mae:.3f}")
print(f"决定系数 (R²): {r2:.3f}")
# model.coef_：模型学到的“权重”
# housing.feature_names：特征的“姓名牌”
# pd.Series(..., index=...)：把两者“缝合”起来
coefs = pd.Series(model.coef_,index=house.feature_names)
print(f"各个项目的影响：{coefs}")
from sklearn.preprocessing import PolynomialFeatures
#X_medinc = X[:,0].reshape(-1,1)# 取出medinc来之后升维进行操作
#poly = PolynomialFeatures(degree=2,include_bias=False)
# degree=2：表示最高次幂为 2
# include_bias=False：表示不生成常数项 1 
#X_poly = poly.fit_transform(X_medinc)
#print("原数据：",X_medinc,"二次项数据：",X_poly)
#plt.scatter(X_medinc, y, color='gray', alpha=0.3, s=10, label='Actual Data (真实数据)')
#plt.plot(X_poly, y_pred, color='red', linewidth=3, label='Polynomial Regression (degree=2)')
#model_linear = LinearRegression()
#model_linear.fit(X_medinc, y)
#y_linear_plot = model_linear.predict(X_poly)
#plt.plot(X_poly, y_linear_plot, color='blue', linestyle='--', label='Linear Regression')
#plt.title('House Price vs Median Income: Linear vs Polynomial', fontsize=14)
#plt.xlabel('Median Income (MedInc)', fontsize=12)
#plt.ylabel('House Price', fontsize=12)
#plt.legend(fontsize=12)
#plt.grid(True, linestyle=':', alpha=0.6)
## 1. 取出 MedInc 特征（全量）
X_medinc = X[:, 0].reshape(-1, 1)

# 2. 生成多项式特征（x 和 x^2）
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_medinc)

# 3. 分别训练两个模型（只用 MedInc 特征！）
# 线性回归模型
model_lin = LinearRegression()
model_lin.fit(X_medinc, y)
y_lin_pred = model_lin.predict(X_medinc)

# 多项式回归模型
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
y_poly_pred = model_poly.predict(X_poly)

# 4. 画图
plt.figure(figsize=(10, 6))
# 画真实数据（灰色散点）
plt.scatter(X_medinc, y, color='gray', alpha=0.3, s=10, label='Actual Data')

# 为了让线条好看，需要对 X_medinc 排序后画线
sort_idx = X_medinc.flatten().argsort()
plt.plot(X_medinc[sort_idx], y_lin_pred[sort_idx], color='blue', linestyle='--', linewidth=2, label='Linear Regression')
plt.plot(X_medinc[sort_idx], y_poly_pred[sort_idx], color='red', linewidth=3, label='Polynomial Regression (degree=2)')

plt.title('House Price vs Median Income: Linear vs Polynomial', fontsize=14)
plt.xlabel('Median Income (MedInc)', fontsize=12)
plt.ylabel('House Price', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()