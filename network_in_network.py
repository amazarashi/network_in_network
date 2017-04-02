import chainer
import chainer.functions as F
import chainer.links as L
import skimage.io as io
import numpy as np
from chainer import utils
import math

class Network_in_Network(chainer.Chain):

    def __init__(self,category_num=10):
        initializer = math.sqrt(2)
        super(Network_in_Network,self).__init__(
            mlp1 = L.MLPConvolution2D(3,(128,96,64),3,stride=2,pad=0,wscale=initializer),
            mlp2 = L.MLPConvolution2D(64,(128,128,128),2,stride=1,pad=0,wscale=initializer),
            mlp3 = L.MLPConvolution2D(128,(128,128,category_num),3,stride=2,pad=0,wscale=initializer)
        )

    def __call__(self,x,train=True):
        #x = chainer.Variable(x)
        h = self.mlp1(x)
        h = F.dropout(h,ratio=.5,train=train)
        h = F.max_pooling_2d(h,3,stride=2,pad=0)

        h = self.mlp2(h)
        h  = F.dropout(h,ratio=.5,train=train)
        h = F.max_pooling_2d(h,3,stride=2,pad=0)

        h = self.mlp3(h)
        h  = F.dropout(h,ratio=.5,train=train)
        h = F.max_pooling_2d(h,3,stride=2,pad=0)

        num, categories, y, x = h.data.shape
        h = F.reshape(F.average_pooling_2d(h,(y, x)), (num, categories))
        return h

    def calc_loss(self,y,t):
        loss = F.softmax_cross_entropy(y,t)
        return loss

    def accuracy_of_each_category(self,y,t):
        y.to_cpu()
        t.to_cpu()
        categories = set(t.data)
        accuracy = {}
        for category in categories:
            supervise_indices = np.where(t.data==category)[0]
            predict_result_of_category = np.argmax(y.data[supervise_indices],axis=1)
            countup = len(np.where(predict_result_of_category==category)[0])
            accuracy[category] = countup
        return accuracy
