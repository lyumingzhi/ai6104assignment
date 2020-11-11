import torch 
import numpy as np 
import math
import random
def relu(input_X):
    return np.where(input_X<0,0,input_X)

def softmax(input_X):
    exp_X=np.exp(input_X)
    sum_exp_X=np.sum(exp_X,axis=1)
    return exp_X/(sum_exp_X.item)
def cross_entropy_error(labels,preds):
    return -np.sum(labels*np.logits(preds))


class Convolution:
    def __init__(self, nf,nc, fh,fw ,stride=1,pad=0):
        W=np.random.randn(nf,nc,fh,fw)*0.01
        self.W=W # fn l k k 
        self.b=None # 1 * fn
        self.stride=stride
        self.pad=pad
        self.col_X=None
        self.X=None
        self.output_size=None
        print('cnn finish initialization')
    def forward(self,input_X):
        self.X=input_X
        nf,nc,fh,fw=self.W.shape

        n,l,xh,xw=self.X.shape
        output_size=(int((2*self.pad+self.X.shape[2]-fh)/self.stride)+1,int((2*self.pad+self.X.shape[3]-fw)/self.stride)+1)

        col_X=self.im2col(fh,fw,self.X)
        exp_x_size=(n*(int((2*self.pad+self.X.shape[2]-fh)/self.stride)+1)*(int((2*self.pad+self.X.shape[3]-fw)/self.stride)+1),fh*fw*nc)
        # print(col_X.shape,exp_x_size,nc)
        assert col_X.shape==exp_x_size

        col_W=self.W.reshape(nf,-1).T
        # print(col_X.shape,col_W.shape)
        # print(np.dot(self.col_X,self.col_W).shape,self.b.repeat(n*[output_size[0]*output_size[1]],axis=0).shape)
        print('w reshape finish')
        # print(col_X[:10,:],col_W.shape)
        output=np.dot(col_X,col_W)
        print('dot finish')
        output=output.T.reshape(nf,n,output_size[0],output_size[1]).transpose((1,0,2,3))
        # output+=self.b
        self.output_size=output.shape
        # print(self.output_size)
        # print(output)
        print('cnn forward finish')
        return output
    def backward(self,dz):
        dz=dz.reshape(self.output_size)
        assert dz.shape==self.output_size
        nf,nc,fh,fw=self.W.shape
        on,ok,oh,ow=dz.shape
        dw_size=self.W.shape
        col_x_z=self.im2col_backward(oh,ow,self.X)
        exp_x_z_size=(self.X.shape[1]*(int((2*self.pad+self.X.shape[2]-oh)/self.stride)+1)*(int((2*self.pad+self.X.shape[3]-ow)/self.stride)+1),oh*ow*on)
        # print(col_x_z.shape,exp_x_z_size)
        assert exp_x_z_size==col_x_z.shape


        col_dz=dz.transpose(1,0,2,3).reshape(ok,-1).T
        self.dw=np.dot(col_x_z,col_dz)
        self.dw=self.dw.T.reshape(ok,nc,dw_size[2],dw_size[3])

        assert self.dw.shape==dw_size

        dx_size=self.X.shape
        col_dz_x=self.im2col(fh,fw,dz)
        col_dz_size=(on*(int((2*self.pad+oh-fh)/self.stride)+1)*(int((2*self.pad+ow-fw)/self.stride)+1),fh*fw*ok)
        assert col_dz_x.shape==col_dz_size

        reversed_W=np.flip(self.W.reshape(nf,nc,-1),-1).reshape(self.W.shape)
        # print('reversed_W',reversed_W)
        col_W=reversed_W.transpose(1,0,2,3).reshape(nc,-1).T
        self.dx=np.dot(col_dz_x,col_W)
        self.dx=self.dx.T.reshape(self.X.shape[1],on,self.X.shape[2],self.X.shape[3]).transpose(1,0,2,3)
        # print(self.dx.shape,self.X.shape)
        assert self.dx.shape==self.X.shape
        # print('w',self.W)
        # print('dz',dz)
        # print('dx',self.dx)
        return self.dx

    def zero_gradient(self):
        self.dw=np.zeros(self.W.shape)

    def step(self,lr):
        self.W-=lr*self.dw



    def im2col_backward(self,fh,fw,X):
        padded_X=np.pad(X,((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)),'constant',constant_values=0).transpose((1,0,2,3))

        new_X=None
        n,l,xh,xw=X.shape
        indexw=indexh=0
        # print('padded_X',padded_X)

        while indexh+fh-1<padded_X.shape[2] :
            indexw=0
            while  indexw+fw-1<padded_X.shape[3]:
                if  new_X is None:
                    new_X=padded_X[:,:,indexh:indexh+fh,indexw:indexw+fw].reshape(l,-1)
                    # print(new_X.shape)
                else:
                    new_X=np.concatenate((new_X,padded_X[:,:,indexh:indexh+fh,indexw:indexw+fw].reshape(l,-1)),axis=1)
                indexw+=self.stride
            indexh+=self.stride
        # print('indexhw',indexh,indexw)
        # print(new_X)
        new_X=new_X.reshape(-1,fh*fw*n)
        # print(new_X.shape)
        return new_X
    def im2col(self,fh,fw,X):
        # print(X)
        padded_X=np.pad(X,((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)),'constant',constant_values=0)
        new_X=None
        n,l,xh,xw=X.shape
        indexw=indexh=0
        # print('padded_X',padded_X)
        while indexh+fh-1<padded_X.shape[2] :
            indexw=0
            while  indexw+fw-1<padded_X.shape[3]:
                if  new_X is None:
                    new_X=padded_X[:,:,indexh:indexh+fh,indexw:indexw+fw].reshape(n,-1)
                    # print(new_X.shape)
                else:
                    new_X=np.concatenate((new_X,padded_X[:,:,indexh:indexh+fh,indexw:indexw+fw].reshape(n,-1)),axis=1)
                indexw+=self.stride
            indexh+=self.stride
        # print('indexhw',indexh,indexw)
        # print(new_X)
        new_X=new_X.reshape(-1,fh*fw*l)
        # print(new_X.shape)
        print('im2col finish')
        # print(X.any()>0)
        # print(new_X.any()>0)
        return new_X

# W=np.array([[[[1,1,1],[1,1,1],[1,1,1]],
#             [[1,1,1],[1,1,1],[1,1,1]],
#             [[1,1,1],[1,1,1],[1,1,1]]],
#             [[[1,1,1],[1,1,1],[1,1,1]],
#             [[1,1,1],[1,1,1],[1,1,1]],
            # [[1,1,1],[1,1,1],[1,1,1]]]])


# W=np.arange((2*1*3*3)).reshape(2,1,3,3)

# # print(W)
# # b=np.ones((1,2,1))
# cnn=Convolution(W,pad=1)
# x=np.arange(2*4*4*1).reshape(2,1,4,4)
# output=cnn.forward(x)
# dz=np.arange(2*2*4*4).reshape(output.shape)
# # print(dz)
# # exit()
# cnn.backward(dz,0.1)
