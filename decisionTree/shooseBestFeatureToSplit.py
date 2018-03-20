#计算给定数据集的香农熵
from math import log
import sys

def calcuXiangNongShang(dataSet):
	num=len(dataSet)
	labelCounts={}
	for featVec in dataSet:
		currentLabel=featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel]=0
		labelCounts[currentLabel]+=1
	shang=0.0
	for key in labelCounts:
		prob=float(labelCounts[key])/num
		shang-=prob*log(prob,2)	
	return shang

def createDataSet():
	dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,0,'no'],[0,1,'no'],[1,0,'yess']]
	labels=['no surfacing','flippers']
	return dataSet,labels

def splitDataSet(dataSet,axis,value):
	retDataSet=[]
	for featVec in dataSet:
		if featVec[axis]==value:
			reduceFeatVec=featVec[:axis]
			reduceFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reduceFeatVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	numFeatures=len(dataSet[0])-1
	print(numFeatures)
	shang=calcuXiangNongShang(dataSet)
	print("原始熵")
	print(shang)
	bestInfo=0.0
	bestFeature=-1
	for i in range(numFeatures):
		featList=[example[i] for example in dataSet]
		uniqueVals=set(featList)
		newShang=0.0
		for value in uniqueVals:
			subDataSet=splitDataSet(dataSet,i,value)
			prob=len(subDataSet)/float(len(dataSet))
			newShang+=prob*calcuXiangNongShang(subDataSet)
		infoGain=shang-newShang
		print("按第"+str(i)+"个特征划分后的熵")
		print(newShang)
		if(infoGain>bestInfo):
			bestInfo=infoGain
			bestFeature=i
	return bestFeature
	

if __name__=='__main__':
	dataSet,labels=createDataSet()
	print(labels)
	print(dataSet)
	print("aaa")
	bestFeature=chooseBestFeatureToSplit(dataSet)
	print(bestFeature)
