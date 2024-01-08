

import numpy as np
import cv2
from sklearn import datasets


#Importing Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

#%%

arr_unit_3d = np.ones((3,2,4), dtype= np.uint8) * 255

#%%
print(arr_unit_3d)

#%%

x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x))
plt.show()

#%%
digits = datasets.load_digits()
#%%
print(digits.data.shape)
print(digits.images.shape)

1#%%
img = digits.images[5, :, : ]
plt.imshow(img, cmap='gray')
plt.show()

#%%

for image_index in range(10):
    subplot_index = image_index +1
    plt.subplot(2,5, subplot_index)
    plt.imshow(digits.images[image_index, :, : ], cmap='gray')
#%%
plt.imshow()
