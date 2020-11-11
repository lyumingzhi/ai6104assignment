import sys ,os 
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf #只是用来加载mnist数据集
from PIL import Image
import pandas as pd 
import math
# x=np.array([[[1,4,4,5],[2,4,5,3]],[[16,18,9,7],[185,5,6,10]],[[26,34,4,127],[182,122,3,0]]])
# print(x.T.shape)
x=np.array([[1,2],[2,3]])
y=np.array([[1,2],[2,3]])
# print(x.transpose(2,1,0))
print(x==y)
# print(x[:,:,1])
# print(np.sum(x,axis=0).reshape(1,-1)+x)
# print(np.flip(x.reshape(1,-1),1).reshape(2,4))
# print(x.shape)
# print(x)
# print(x.transpose(1,0,2))
# print(np.pad(x,((0,0),(1,1),(1,1)),'constant'))