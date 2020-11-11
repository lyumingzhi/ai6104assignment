import numpy as np 
class Pooling:
    def __init__(self,ph,pw,stride=1,pad=0):
        self.ph=ph
        self.pw=pw
        self.stride=stride
        self.pad=pad
        self.X=None
        assert stride==ph
        print('pl finish initialization')
    def forward(self,X):
        self.X=X
        n,c,h,w=self.X.shape
        col_X=self.im2col(self.ph,self.pw,self.X)
        col_X=col_X.reshape(-1,self.ph*self.pw)
        self.argmax=np.argmax(col_X,axis=-1)

        self.out_shape=(int((self.X.shape[2]+2*self.pad-self.ph)/self.stride+1),int((self.X.shape[3]+2*self.pad-self.pw)/self.stride+1))
        # print(self.out_shape)
        out=np.max(col_X,axis=-1).reshape(n,self.out_shape[0],self.out_shape[1],c).transpose(0,3,1,2)
        return out
    def backward(self,dz):
        # dx=np.zeros(self.X.shape)
        dz=dz.reshape(self.X.shape[0],self.X.shape[1],self.out_shape[0],self.out_shape[1])
        n,c,h,w=self.X.shape
        col_dx=np.zeros((n*c*self.out_shape[0]*self.out_shape[1],self.ph*self.pw))
        col_dx[np.arange(self.argmax.size),self.argmax.flatten()]=dz.flatten()
        self.dx=self.col2im(self.ph,self.pw,col_dx)
        assert self.X.shape==self.dx.shape
        return self.dx

    def col2im(self,fh,fw,colX):
        n,c,h,w=self.X.shape
        imx=np.zeros(self.X.shape)
        padded_X=np.pad(imx,((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)),'constant',constant_values=0)
        indexw=indexh=indexn=indexc=0
        indexow=indexoh=0
        colX=colX.reshape(n,c,self.out_shape[0],self.out_shape[1],self.ph,self.pw)

        # print('colX',colX.shape,'padded_X',padded_X.shape)
        while indexh+fh-1<padded_X.shape[2] :
            indexw=0
            indexow=0
            while  indexw+fw-1<padded_X.shape[3]:
                padded_X[:,:,indexh:indexh+fh,indexw:indexw+fw]=colX[:,:,indexoh,indexow,:,:]
                indexow+=1
                indexw+=self.stride
            indexoh+=1
            indexh+=self.stride
            
        return padded_X[:,:,self.pad:h-self.pad,self.pad:w-self.pad]

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
        return new_X



# x=np.arange(2*2*4*4).reshape(2,2,4,4)
# print('x',x)
# pooling=Pooling(2,2,2,0)
# print('forward',pooling.forward(x))

# dz=np.ones((2,2,2,2))
# print('backward',pooling.backward(dz))