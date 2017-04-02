import argparse
from chainer import optimizers
from chainer import serializers
import squeeze
import amaz_tester
import amaz_cifar10_dl
import amaz_augumentationCustom
import amaz_optimizer
import cv2
import numpy as np

if __name__ == '__main__':
    x_path = "/Users/suguru/Desktop/dog2.jpg"
    img = cv2.imread(x_path)
    img = np.asarray(img).transpose(2,0,1).astype(np.float32)/255.
    model = squeeze.Squeeze(10)
    dataset = amaz_cifar10_dl.Cifar10().simpleLoader()
    dataaugumentation = amaz_augumentationCustom.Normalize128
    test = amaz_tester.Tester(model,dataset,dataaugumentation)
    res = test.executeOne(img)
    print(res)
