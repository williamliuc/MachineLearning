#coding: utf-8
from numpy import *
import re

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
			#returnVec[vocabList.index(word)]=1#词集模型
			returnVec[vocabList.index(word)]+=1#词袋模型
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

def textParse(bigString):
	listOfTokens=re.split('\W',bigString)#分隔符是除单词，数字外的任意字符串
	return [tok.lower() for tok in listOfTokens if len(tok)>2]#去掉空字符串，所有单词小写

def spamTest():
	docList=[];classList=[];fullText=[]
	for i in range(1,26):
		wordList=textParse(open('email/spam/%d.txt'%i).read())
		docList.append(wordList)#使用append的时候，是将wordList看作一个对象，整体打包添加到docList对象中
		fullText.extend(wordList)#使用extend的时候，是将wordList看作一个序列，将这个序列和fullText序列合并，并放在其后面
		classList.append(1)
		wordList=textParse(open('email/ham/%d.txt'%i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)

	vocabList=createVocabList(docList)
	trainingSet=list(range(50));testSet=[]
	for i in range(10):
		randIndex=int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat=[];trainClasses=[]
	print(len(trainingSet))
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0v,p1v,pAb=trainNBO(trainMat,trainClasses)
	errorCount=0
	for docIndex in testSet:
		wordVector=setOfWords2Vec(vocabList,docList[docIndex])
		if classifyNB(wordVector,p0v,p1v,pAb)!=classList[docIndex]:
			errorCount+=1
	print('the error rate is :',float(errorCount)/len(testSet))

if __name__=='__main__':
	spamTest()
