#完整的决策树代码，包括计算香农熵，创建数据集，根据特征值提取数据集，选择最优划分的最优特征等函数
from math import log
import sys
import operator

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
	dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,0,'no'],[0,1,'no']]
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
	
def majorityCnt(classList):
	classCount={}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote]=0
		classCount[vote]+=1
	sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

def createTree(dataSet,labels):
	classList=[example[-1] for example in dataSet]
	print(classList)
	if classList.count(classList[0])==len(classList):
		return classList[0]
	if len(dataSet[0])==1:
		return majorityCnt(classList)
	bestFeat=chooseBestFeatureToSplit(dataSet)
	bestFeatLabel=labels[bestFeat]
	myTree={bestFeatLabel:{}}
	print(labels)
	print(bestFeat)
	del(labels[bestFeat])
	featValues=[example[bestFeat] for example in dataSet]
	uniqueVals=set(featValues)
	for value in uniqueVals:
		subLabels=labels[:]#python中传递列表是传递引用，为了不让后面的递归函数改变这里labels中的值，我们复制一份，传递sublabels
		print(labels)
		#sys.exit(0)
		myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
	return myTree

if __name__=='__main__':
	dataSet,labels=createDataSet()
	myTree=createTree(dataSet,labels)
	print(myTree)
