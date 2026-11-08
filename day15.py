import numpy as np
import matplotlib.pyplot as plt
# shape 表示数组形状，如shape[0]表示第0个维度的大小
# shape 里的每个数字，表示对应轴（axis）上有多少个元素
# 0 维是标量：没有轴，只有一个元素；1 维是向量/序列：有一个轴，长度可以是 n，也可以是 0 或 1
# shape[i]：第 i 个轴的长度。二维矩阵中，shape[0] 是行数，shape[1] 是列数
# 因为一维数组只有一个轴，即 axis=0
#x_train = np.array([1.0,2.0])
#y_train = np.array([300,500])
#print(x_train.shape[0])
#plt.figure()
#plt.scatter(x_train,y_train,marker='x',c='r')
# s：点的大小（面积，points²），标量或数组
# c：点的颜色，颜色字符串、颜色序列或数值数组（配合 cmap）
# marker：点的形状，默认 'o'
#plt.title("House price")
#plt.xlabel('Size')
#plt.ylabel('Price(in 1000s of dollars)')
w = 100
b = 100
#m = len(x_train)
i = 0
#def calculate_model(w, b, x):
    # 直接对整个数组进行乘法和加法，不需要写 for 循环
# 调用函数
#y_pred = calculate_model(w, b, x_train)
# 绘图
#plt.plot(x_train, y_pred, c='g', label='Prediction')
# 如果没有在 plot 中写 label，plt.legend() 要么什么都不显示，要么报警告
def compute_cost(x,y,w,b):
    m=x.shape[0] 
    cost_sum=0  
    for i in range(m):
       f_wb = w * x[i] + b  
       cost = (f_wb - y[i]) ** 2 
       cost_sum = cost_sum + cost 
       total_cost = (1 / (2 * m)) * cost_sum  
    return total_cost
#plt.legend()
#plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
np.random.seed(42)
X = np.arange(1,11).reshape(-1,1)
y = 2*X.ravel() + 3 + np.random.rand(10)*0.5 # ravel 的意思是“铺平、展平”-降维
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 2  8划分训练集和测试集 
# 种子：是伪随机数生成器的初始密码，决定了随机序列的起点。
# 作用：保证实验可复现，方便调试，排除“运气”因素
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
# fit 就是“训练”、“学习”的意思。这行代码是在对模型说：“嘿，这是特征数据（X_train），这是对应的真实答案（y_train）
mse = mean_squared_error(y_test, y_pred)
print(f"模型的斜率（权重）: {model.coef_[0]:.2f}")
# model.coef_：这是线性回归模型在 fit() 训练后自动生成的属性，代表权重系数（w）
# [0]：注意这里的中括号。因为机器学习模型通常处理多维特征，coef_ 返回的是一个数组（哪怕你只有一个特征）。加上 [0] 表示取出第一个特征对应的权重。
print(f"模型的截距: {model.intercept_:.2f}")
# model.intercept_：训练后生成的属性，代表截距（b）
print(f"均方误差 MSE: {mse:.2f}")
plt.scatter(X, y, c='b', label='ture datas')
plt.plot(X, model.predict(X), color='red', label='predict line')
plt.xlabel('area')
plt.ylabel('price')
plt.title('housing price example')
plt.legend()
plt.show()