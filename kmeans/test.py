# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 12:30:11 2017

@author: chao
"""
from numpy import *  
import kmeans 
  
## 读数据
print "step 1: load data..."  
dataSet = []  
fileIn = open('/home/chao/Desktop/python_work/kmeans/test.txt')  
for line in fileIn.readlines():  
    lineArr = line.strip().split(',')  
    dataSet.append([float(lineArr[0]), float(lineArr[1])]) #将每一组数据读入列表里面 
  
## 聚类
print "step 2: clustering..."  
dataSet = mat(dataSet) #mat函数创建矩阵
k = 4
centroids, clusterAssment = kmeans.kmeans(dataSet, k)  
## 画出结果图
print "step 3: show the result..."  
kmeans.showCluster(dataSet, k, centroids, clusterAssment)