#完整的决策树代码，包括计算香农熵，创建数据集，根据特征值提取数据集，选择最优划分的最优特征等函数
#此决策树解决的问题：海洋动物中，根据不浮出水面是否可以生存，以及是否有脚蹼。我们可以将这些动物分成两类：鱼类和非鱼类。
#createDataSet()函数中给定了一些实际海洋生物数据，简而言之此决策树的作用就是要在这些生物数据中根据决策树算法自动学习形成一棵规则树。
#当有新数据来的时候可以根据这棵规则树自动将海洋生物分类，而不需要人工判断。当然这里数据量很小，规则也很少，我们人类都可以瞬间学习然后
#判断，但是当海量数据和海量规则的时候就很难了。
from math import log
import sys
import operator

#创建数据集，返回值：数据集，类标签（特征的表现）yes：鱼类，no：非鱼类，1：是，0：否
def createDataSet():
	dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,0,'no'],[0,1,'no']]
	labels=['no surfacing','flippers']
	return dataSet,labels

#计算香农熵
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

#根据指定特征值在原数据集中提取新的数据集，axis：特征的下标，表示第几个特征。value：特征值；返回值：新的数据集
def splitDataSet(dataSet,axis,value):
	retDataSet=[]
	for featVec in dataSet:
		if featVec[axis]==value:
			reduceFeatVec=featVec[:axis]
			reduceFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reduceFeatVec)
	return retDataSet

#选择最优划分的最优特征下标，返回值：最优划分的特征值得下标
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

#选出一个集合中占大多数的类别标签，返回值：类别标签
def majorityCnt(classList):
	classCount={}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote]=0
		classCount[vote]+=1
	sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

#创建决策树的递归算法
def createTree(dataSet,labels):
	classList=[example[-1] for example in dataSet]
	if classList.count(classList[0])==len(classList):#类别完全相同停止继续划分
		return classList[0]
	if len(dataSet[0])==1:
		return majorityCnt(classList)#遍历所有特征时返回出现次数最多的
	bestFeat=chooseBestFeatureToSplit(dataSet)
	bestFeatLabel=labels[bestFeat]
	myTree={bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues=[example[bestFeat] for example in dataSet]#得到列表包含的所有属性值
	uniqueVals=set(featValues)
	for value in uniqueVals:
		subLabels=labels[:]#python中传递列表是传递引用，为了不让后面的递归函数改变这里labels中的值，我们复制一份，传递sublabels
		myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
	return myTree

if __name__=='__main__':
	dataSet,labels=createDataSet()
	myTree=createTree(dataSet,labels)
	print(myTree)
