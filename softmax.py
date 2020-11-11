import numpy as np 

def softmax(X):
    exp_X=np.exp(X)
    sum_X=np.sum(exp_X,axis=1)
    sum_X=sum_X.reshape(X.shape[0],-1)
    output=exp_X/sum_X
    return output

class Softmax:
    def __init__(self):
        self.y_hat=None
    def forward(self,X):
        self.y_hat=softmax(X)
        return self.y_hat
    def backward(self,label):
        self.dx=label-self.y_hat
        return self.dx

# x=np.array([[0,1,0]])
# sm=Softmax()
# print(sm.forward(x))