import matplotlib.pyplot as plt
import sys, os
from  PIL import Image
import math
import pandas as pd
import torch
from utils import check_mnist_dataset_exists
import cnn
import fc
import softmax
import crossentropy
import relu
import pooling
import Model
import numpy as np
def load_data():
    data_path=check_mnist_dataset_exists()

    train_data=torch.load(data_path+'mnist/train_data.pt')
    train_label=torch.load(data_path+'mnist/train_label.pt')
    test_data=torch.load(data_path+'mnist/test_data.pt')
    test_label=torch.load(data_path+'mnist/test_label.pt')
    return train_data.numpy(),train_label.numpy(),test_data.numpy(),test_label.numpy()

# load_data()
def one_hot_encoding(labels):
    new_labels=np.zeros((labels.shape[0],10))
    # for i in range(labels.shape[0]):
    new_labels[range(labels.flatten().shape[0]),labels]=1
    return new_labels


def train(model,criterion,train_data,train_label):
    lr=0.01
    bs=10
    for epoch in range(200):
        average_loss=0
        shuffled_indices=torch.randperm(train_data.shape[0])

        for index in range(0,train_data.shape[0],bs):
            indics=shuffled_indices[index:index+bs]
            data=train_data[indics,:,:,:]
            labels=train_label[indics,:]
            preds=model.forward(data)
            loss=criterion.forward(preds,labels)
            dloss=criterion.backward()
            print(dloss.shape)

            model.zero_gradient()
            model.backward(dloss)
            model.step(lr)
            average_loss+=loss
            print(np.sum(loss,axis=-1))
        average_loss/=train_data.shape[0]/bs
def predict(model,test_data,test_label):
    bs=10
    average_accuracy=0
    for index in range(0,test_data.shape[0],bs):
        data=test_data[index:index+bs,:,:,:]
        labels=test_label[index:index+bs,:]
        preds=model.forward(data)
        preds=np.argmax(preds,axis=1)
        labels=np.argmax(labels,axis=1)
        accuracy=np.sum(preds==labels,axis=0).reshape(1,)[0]/bs
        print('accuracy: ',accuracy)
        average_accuracy+=accuracy
    average_accuracy/=test_data.shape[0]/bs
    print('average_accuracy:',average_accuracy)



if __name__=='__main__':
    train_data,train_label,test_data,test_label=load_data()
    train_data=train_data.reshape(train_data.shape[0],1,train_data.shape[1],train_data.shape[2])
    test_data=test_data.reshape(test_data.shape[0],1,test_data.shape[1],test_data.shape[2])
    train_label=one_hot_encoding(train_label)
    test_label=one_hot_encoding(test_label)
    print(train_data.shape)
    criterion=crossentropy.CrossEntropy(10)
    model=Model.Model([cnn.Convolution(50,1,3,3,1,1),
                        relu.Relu(),
                        cnn.Convolution(100,50,3,3,1,1),
                        pooling.Pooling(2,2,2,0),
                        relu.Relu(),
                        pooling.Pooling(2,2,2,0),
                        # fc.FC(int((train_data.shape[0]/2/2)*(train_data.shape[0]/2/2)*10),100),
                        fc.FC(4900,100),
                        relu.Relu(),
                        fc.FC(100,10)]
                        )
    print('i am here')
    train(model,criterion,train_data,train_label)
    print(train_data.shape,train_label.shape,test_data.shape)
