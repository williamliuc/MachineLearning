#选择最优的特征来进行决策树的划分，使其熵最小
from math import log
import sys

#计算给定数据集的香农熵
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

#创建数据集，数据集必须为列表，所有列表实例都要具有相同的数据长度，每个实例的最后一个元素是当前实例的类别标签
def createDataSet():
	dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,0,'no'],[0,1,'no'],[1,0,'yess']]
	labels=['no surfacing','flippers']
	return dataSet,labels

#选取集合中第axis个特征等于value的所有实例（元素），形成一个子列表retDataSet
def splitDataSet(dataSet,axis,value):
	retDataSet=[]
	for featVec in dataSet:
		if featVec[axis]==value:
			reduceFeatVec=featVec[:axis]
			reduceFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reduceFeatVec)
	return retDataSet

#遍历实例中的每一个特征，将每一个特征划分后的集合的熵进行比较，选取熵最小的划分，返回决定熵最小划分的特征值的序号
def chooseBestFeatureToSplit(dataSet):
	numFeatures=len(dataSet[0])-1#获取集合中特征个数
	shang=calcuXiangNongShang(dataSet)
	print("原始熵")
	print(shang)
	bestInfo=0.0
	bestFeature=-1
	for i in range(numFeatures):#遍历每一个特征
		featList=[example[i] for example in dataSet]
		uniqueVals=set(featList)#统计第i个特征的不同值
		newShang=0.0
		for value in uniqueVals:#统计第i个特征划分后集合的熵，存入newShang。比如此特征将集合划分为两类，划分后的熵=第一类的熵*第一类元素比例+第二类的熵*第二类元素比例
			subDataSet=splitDataSet(dataSet,i,value)
			prob=len(subDataSet)/float(len(dataSet))
			newShang+=prob*calcuXiangNongShang(subDataSet)
		infoGain=shang-newShang
		print("按第"+str(i)+"个特征划分后的熵")
		print(newShang)
		if(infoGain>bestInfo):#选择最小的划分后的熵
			bestInfo=infoGain
			bestFeature=i
	return bestFeature #返回最好特征的序号
	

if __name__=='__main__':
	dataSet,labels=createDataSet()
	print(labels)
	print(dataSet)
	bestFeature=chooseBestFeatureToSplit(dataSet)
	print(bestFeature)
