#计算给定数据集的香农熵
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
