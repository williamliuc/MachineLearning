# -*- coding: utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt

#计算两向量之间的欧式距离，在这里是计算两点之间的距离
def euclDistance(vector1,vector2):
    return sqrt(sum(power(vector2-vector1,2)))

#初始化......
#从原始数据中产生随机的k个数据存入centroids
def initCentroids(dataSet,k):
    numSamples,dim=dataSet.shape#返回dataSet的行和列
    centroids=zeros((k,dim))#创建k行dim列的矩阵
    for i in range(k):
        index=int(random.uniform(0,numSamples))#从0到numSamples中随机产生一个数
        centroids[i,:]=dataSet[index,:]
    return centroids

def kmeans(dataSet,k):#此算法用到3个数据集，dataSet:n行两列表示原始数据，clusterAssment:n行两列，第一列表示
                      #原始数据的类型，第二列表示此点到质心的距离，centriods:k行两列表示点群的质心
    numSamples=dataSet.shape[0]
    clusterAssment=mat(zeros((numSamples,2)))#clusterAssment中存放点聚类的类别以及与该类别质心的距离
    clusterChanged=True
    centroids=initCentroids(dataSet,k)#从原始数据中产生随机的k个数据存入centroids，代表k个质心
    while clusterChanged:
        clusterChanged=False
        for i in xrange(numSamples):
            minDist=100000.0
            minIndex=0
            for j in range(k):#从k个质心中选取距离i行这个点最小的一个质心
                distance=euclDistance(centroids[j,:],dataSet[i,:])
                if distance<minDist:
                    minDist=distance
                    minIndex=j
            
            if clusterAssment[i,0]!=minIndex:
                clusterChanged=True#直到对于所有的原始数据类别都确定，都不再更新，即
                                   #（所有的clusterAssment[i,0]都等于minIndex）。此标志为false，退出while循环
                clusterAssment[i,:]=minIndex,minDist**2
                
        for j in range(k):#更新每个点群的质心
            pointsInCluster=dataSet[nonzero(clusterAssment[:,0]==j)[0]]#选取j类的所有点存入pointsInCluster，这里nonzero函数是个难点，可以百度一下
            centroids[j,:]=mean(pointsInCluster,axis=0)#对pointInCluster中的数据按列求均值
            

    #kmeans算法不包括这里的代码，这里的代码主要是可以打印清楚质心的移动情况      
        mark=['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        #画聚类后的图  
        for i in xrange(numSamples):
            markIndex=int(clusterAssment[i,0])
            plt.plot(dataSet[i,0],dataSet[i,1],mark[markIndex],markersize=6)
        mark=['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    #画质心
        for i in range(k):
            plt.plot(centroids[i,0],centroids[i,1],mark[i],markersize=12)
        plt.show()
        

    print "聚类完成"
    return centroids,clusterAssment

def showCluster(dataSet,k,centroids,clusterAssment):
    numSamples,dim=dataSet.shape
    if dim!=2:
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1
    mark=['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k>len(mark):
        print "Sorry! Your k is too large! please contact Zouxy"
        return 1
    #画聚类后的图  
    for i in xrange(numSamples):
        markIndex=int(clusterAssment[i,0])
        plt.plot(dataSet[i,0],dataSet[i,1],mark[markIndex],markersize=6)
    mark=['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    #画质心
    for i in range(k):
        plt.plot(centroids[i,0],centroids[i,1],mark[i],markersize=12)
    plt.show()
       
            
            
            
            
            
            
            
            
            
            
            
            
            
            