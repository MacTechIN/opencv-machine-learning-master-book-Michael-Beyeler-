
import numpy as np


y_true = np.random.randint(0,2,size=5)

print("y_true: ", y_true)

y_pred = np.ones(5, dtype= np.int32)
print("y_pred: ", y_pred)

print(np.sum(y_true == y_pred) / len(y_true))

#%%
from sklearn import metrics

result = metrics.accuracy_score(y_pred, y_true)

print("metrics accuracy result:" , result )


#%%

x = np.linspace(0, 10, 100)

y_true = np.sin(x) + np.random.rand(x.size) - 0.5

y_pred = np.sin(x)

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.plot(x, y_pred, linewidth=4 , label= 'model')
plt.plot(x, y_true, 'o', label='data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower left')

plt.show()

#%%

mse = np.mean((y_true - y_pred)**2)
print(mse)

#%%
##사이킷런에서 MSE 를 계산 하기 쉽다 (결과 값. 예측갑  이용)

print( "mse by sklearn : " , metrics.mean_squared_error(y_true, y_pred))


#%%
# fvu  : fraction of variance unexplaied  설명할 수 없는 분산 분율

fvu = np.var(y_true-y_pred) / np.var(y_true)
print(fvu)

#%%
fve = 1.0 - fvu
print(fve)

#%%
# Scikit learn 으로 다시 풀어보면

fve2 = metrics.explained_variance_score(y_true, y_pred)
print("fev : ", fve2)

r2 = metrics.r2_score(y_true, y_pred)
print("r2 score : ", r2)

#%%

pred_rate =  metrics.r2_score(y_true, y_pred * np.ones_like(y_true))
print( "predit accuracy rate : ", pred_rate)