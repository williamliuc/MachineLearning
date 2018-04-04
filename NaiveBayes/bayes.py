此代码非常简单，但是它展示了朴素贝叶斯分类器的工作原理
from numpy import *

def loadDataSet():
	postingList=[['my','dog','flea','problems','help','please'],['maybe','not','take',\
	'him','to','dog','park','stupid'],['my','dalmation','is','so','cute','I','love',\
	'him'],['stop','posting','stupid','worthless','garbage'],['mr','licks','ate','my','steak',\
	'how','to','stop','him'],['quit','buying','worthless','dog','food','stupid']]
	classVec=[0,1,0,1,0,1]	#1代表侮辱性文字，0代表正常言论
	return postingList,classVec

#获取所有文档的词汇表，词汇表：dataSet中的所有文档列表中出现的不重复词汇的一个列表集合
def createVocabList(dataSet):
	vocabSet=set([])
	for document in dataSet:
		vocabSet=vocabSet|set(document)#符号|取两个集合的并集,顺序随机
	return list(vocabSet)

#输入：vocabList：词汇表，inputSet：某个文档
#获取某一文档的文档向量，文档向量：与词汇表等长，每一元素为1或0，分别表示词汇表中的单词在输入文档中是否出现
def setOfWords2Vec(vocabList,inputSet):
	returnVec=[0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)]=1
		else: print("the word:",word,"is not in my vocabulary!")
	return returnVec
def trainNBO(trainMatrix,trainCategory):
	numTrainDocs=len(trainMatrix)
	numWords=len(trainMatrix[0])
	print(numWords)
	pAbusive=sum(trainCategory)/float(numTrainDocs)
	print(pAbusive)
	#p0Num=zeros(numWords);p1Num=zeros(numWords)#为了避免概率中0的出现导致最终所算的概率为0，我们将分子设为1，分母设为2
	#p0Denom=0.0;p1Denom=0.0
	p0Num=ones(numWords);p1Num=ones(numWords)
	p0Denom=2.0;p1Denom=2.0
	for i in range(numTrainDocs):
		if trainCategory[i]==1:
			p1Num+=trainMatrix[i]
			p1Denom+=sum(trainMatrix[i])
		else:
			p0Num+=trainMatrix[i]
			p0Denom+=sum(trainMatrix[i])
	#p1Vect=p1Num/p1Denom#python中如果有很多非常小的数相乘，那么结果会为0，为了避免这种情况采用取对数的方法。
	#p0Vect=p0Num/p0Denom
	p1Vect=log(p1Num/p1Denom)
	p0Vect=log(p0Num/p0Denom)
	return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
	p1=sum(vec2Classify*p1Vec)+log(pClass1)
	p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
	if p1>p0:
		return 1
	else:
		return 0

if __name__=='__main__':
	listOPosts,classVec=loadDataSet()
	#print(postingList,classVec)
	myVocabList=createVocabList(listOPosts)
	print(myVocabList)
	trainMat=[]
	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
	print(trainMat)
	p0v,p1v,pAb=trainNBO(trainMat,classVec)
	
	#测试
	testEntry1=['love','my','dalmation']
	thisDoc=setOfWords2Vec(myVocabList,testEntry1)
	print('testEntry1 class is:',classifyNB(thisDoc,p0v,p1v,pAb))
	testEntry2=['stupid','garbage']
	thisDoc=setOfWords2Vec(myVocabList,testEntry2)
	print('testEntry2 class is:',classifyNB(thisDoc,p0v,p1v,pAb))
