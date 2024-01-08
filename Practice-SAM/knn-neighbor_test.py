# Red team, Blue team 예측 하기
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import array

#%%
plt.style.use('ggplot')

#%%
# Train Data 만들기
np.random.seed(42)

#%%
# Single Data point

single_data_point = np.random.randint(0,100, 2)
print('single_data_pint:', single_data_point)
#%%
single_label = np.random.randint(0,2)
print("single_label:", single_label)
#0 = blue square , 1 = red tringle

#%%
def generate_data(num_samples, num_feature= 2):
    data_size =  (num_samples, num_feature)
    data = np.random.randint(0,100, size= data_size)

    label_size = (num_samples, 1)
    labels = np.random.randint(0,2,size= label_size)

    return data.astype(np.float32), labels

#%%
train_data , labels = generate_data(11)

#%%

print(train_data[0])
print(labels[0])


#%%
plt.plot(train_data[0,0], train_data[0,1], "sb")
plt.show()


#%%
def plot_data(all_blue, all_red):
    plt.scatter(all_blue[:, 0], all_blue[:,1] , c='b', marker='s', s=180)
    plt.scatter(all_red [:,0], all_red[:,1],  c='r',  marker='^', s=180)

    plt.xlabel('x coord (feature 1)')
    plt.ylabel('y coord (feature 2)')

    plt.show()

#%%
# Blue, Red 데이터 셋 분리
labels.ravel() == 0
array([True, True, False, False, False, True, False, True, True, True ], dtype= bool)


blue = train_data[labels.ravel() == 0]
red = train_data[labels.ravel() == 1]

#%%
plot_data(blue, red)


#%%
knn = cv2.ml.KNearest_create()
#%%
knn.train(train_data, cv2.ml.ROW_SAMPLE, labels)


#%%
newcomer , _ = generate_data(1)
print(newcomer)

#%%
plot_data(blue, red)
plt.plot(newcomer[0,0], newcomer[0,1], 'go', markersize= 14);

#%%

ret, results, neighbor, dist = knn.findNearest(newcomer, 1)
print("Predicted label:\t", results)
print("Neighbor's label:\t", neighbor)
print("Distance to neighbor:\t", dist)

#%%
ret, results, neighbors, dist = knn.findNearest(newcomer, 7)
print("Predicted label:\t", results)
print("Neighbor's label:\t", neighbors)
print("Distance to neighbor:\t", dist)
