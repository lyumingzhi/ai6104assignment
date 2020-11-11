import numpy as np 
class Relu:
	def __init__(self):
		self.mask=None
		print('relu finish initialization')
	def forward(self,X):
		self.mask=X<=0
		X[self.mask]=0

		return X
	def backward(self,dz):
		dz[self.mask]=0
		return dz

# relu=Relu()
# x=np.arange(-4,5).reshape(3,3)
# print(relu.forward(x))