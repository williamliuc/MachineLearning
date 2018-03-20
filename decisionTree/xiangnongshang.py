#计算给定数据集的香农熵,香农熵：个人理解为一个数据集当中数据之间的混合程度，也可以说是要将这个数据集根据不同特征划分为不同类的难度，
#熵越高代表着集合中混合数据的类别越多，划分难度越大。
#简便计算方式（个人猜测加总结）：如果一个集合可以划分为n（n为2,4,6,8,16...）类，那么熵就是n的以2为底的对数。例：如果一个集合可以划分为8类，熵就等于3
#标准计算公式：这里不好写，百度搜
from math import log

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
	print(shang)

def createDataSet():
	dataSet=[[1,1,'ha89'],[1,1,'ha7'],[1,0,'ha8'],[0,1,'ha3'],[0,1,'ha17'],[1,2,'ha18'],[1,3,'ha2'],[1,5,'ha26a'],[1,1,'ha89a'],[1,1,'ha72a'],[1,0,'ha823a'],[0,1,'ha33a'],[0,1,'ha173a'],[1,2,'ha182a'],[1,3,'ha22a'],[1,5,'ha261a']]
	labels=['no surfacing','flippers']
	return dataSet,labels

if __name__=='__main__':
	dataSet,labels=createDataSet()
	print(labels)
	print(dataSet)
	calcuXiangNongShang(dataSet)
