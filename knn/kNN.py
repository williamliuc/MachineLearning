# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from numpy import *
import os
def kNNClassify(newInput,dataset,labels,k):#knn算法核心
    #先求数据库中每个点到所求点之间的距离
    numSamples=dataset.shape[0] #获取数据库的行数
    diff=tile(newInput,(numSamples,1))-dataset#使用tile函数迭代创建一个numSample行1列的array与dataset做差
    squaredDiff=diff**2#diff中的每一个数都平方
    squaredDist=sum(squaredDiff,axis=1)#每一行的数求和
    distance=squaredDist**0.5#在开方
    #再对距离进行排序
    sortedDistIndices=argsort(distance)
    classCount={}
    #统计距离为k的每个点的类别
    for i in xrange(k):
       voteLabel=labels[sortedDistIndices[i]]
       classCount[voteLabel]=classCount.get(voteLabel,0)+1
    maxCount=0              
    #找出离所求点最近的k个点中最多的类别         
    for key,value in classCount.items():
        if maxCount<value:
            maxCount=value
            maxIndex=key
    #返回所求点的类型，算法到此结束
    return maxIndex
 
def img2vector(filename):
    rows=32
    cols=32
    imgVector=zeros((1,rows*cols))
    fileIn=open(filename)
    for row in xrange(rows):
        lineStr=fileIn.readline()
        for col in xrange(cols):
            imgVector[0,row*32+col]=int(lineStr[col])
    return imgVector

def loadDataSet():
    dataSetDir='C:/knn/'
    trainingFileList=os.listdir(dataSetDir+'trainingDigits')
    numSamples=len(trainingFileList)
    train_x=zeros((numSamples,1024))
    train_y=[]
    for i in xrange(numSamples):
        filename=trainingFileList[i]
        train_x[i,:]=img2vector(dataSetDir+'trainingDigits/%s'%filename)
        label=int(filename.split('_')[0])
        train_y.append(label)
    testingFileList=os.listdir(dataSetDir+'testDigits')
    numSamples=len(testingFileList)
    test_x=zeros((numSamples,1024))
    test_y=[]
    for i in xrange(numSamples):
        filename=testingFileList[i]
        test_x[i,:]=img2vector(dataSetDir+'testDigits/%s'%filename)
        label=int(filename.split('_')[0])
        test_y.append(label)
    return train_x,train_y,test_x,test_y

def testHandWritingClass():
    print "第一步：加载数据。。。"
    train_x,train_y,test_x,test_y=loadDataSet()
   
    numTestSamples=test_x.shape[0]
    print "数据加载完成"
    matchCount=0
    for i in xrange(numTestSamples):
        print i
        predict=kNNClassify(test_x[i],train_x,train_y,9)
        if predict==test_y[i]:
            matchCount+=1
    accuracy=float(matchCount)/numTestSamples
    
    print "accuracy is:%.2f%%"%(accuracy*100)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    