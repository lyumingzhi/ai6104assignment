import numpy as np 
# import random
class FC:
	def __init__(self,input_size,output_size):
		self.W=np.random.randn(input_size,output_size)*0.01
		self.b=np.zeros((1,output_size))
		self.X=None 
		self.dW=None
		self.db=None
		self.original_size_X=None
		print('fc finish initialization')
	def forward(self,X):
		self.original_size_X=X.shape
		self.X=X.reshape(X.shape[0],-1)
		output=np.dot(self.X,self.W)+self.b
		self.output_size=output.shape
		return output
	def backward(self,dz):
		
		dz=dz.reshape(self.X.shape[0],-1)
		self.dw=np.dot(self.X.T,dz)/self.X.shape[0]
		self.db=np.sum(dz,axis=0)/self.X.shape[0]
		self.dx=np.dot(dz,self.W.T)
		return self.dx
		print('w',self.W)
		print('dz',dz)
		print('dx',self.dx)
	def step(self,lr):
		self.lr=lr
		self.W-=lr*self.dw 
		self.b-=lr*self.db
	def zero_gradient(self):
		self.dw=np.zeros(self.W.shape)
		self.db=np.zeros(self.b.shape)

# W=np.arange(3*4).reshape(3,4)
# b=np.arange(4).reshape(1,-1)
# fc=FC(W,b)
# x=np.arange(3*3).reshape(3,3)
# print(fc.forward(x))
# dz=np.arange(3*4).reshape(3,4)
# fc.backward(dz,0.1)