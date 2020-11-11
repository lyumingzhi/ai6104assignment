import numpy as np 
class CrossEntropy(object):
	"""docstring for CrossEntropy"""
	def __init__(self, numC):
		super(CrossEntropy, self).__init__()
		self.numC=numC
		print('ce finish initialization')
	def forward(self,X,labels):
		self.softmax_X=self.softmax(X)
		# print(self.softmax_X)
		self.labels=labels

		self.loss=np.sum(self.softmax_X*labels,axis=-1)/X.shape[0]
		return self.loss
	def backward(self):
		self.dx=self.softmax_X-self.labels
		return self.dx
	def softmax(self,X):
	    exp_X=np.exp(X)
	    # print(exp_X)
	    sum_X=np.sum(exp_X,axis=-1)
	    # print(sum_X)
	    sum_X=sum_X.reshape(X.shape[0],-1)
	    output=exp_X/sum_X
	    return output


# x=np.arange(2*4).reshape(2,4)
# print(x)
# y=np.array([[1,0,0,0],[0,1,0,0]])
# loss=CrossEntropy(x.shape[-1])
# print(loss.forward(x,y))
# print(loss.backward())