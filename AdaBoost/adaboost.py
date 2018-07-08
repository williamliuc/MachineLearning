from numpy import *
def loadSimpData():
	datMat = matrix([[1,2.1],[2,1.1],[1.3,1],[1,1],[2,1]])
	classLabels = [1,1,-1,-1,1]
	return datMat,classLabels

#通过阈值比较对数据进行分类
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
	retArray = ones((shape(dataMatrix)[0],1))
	if threshIneq == 'lt':
		retArray[dataMatrix[:,dimen] <= threshVal] = -1
	else:
		retArray[dataMatrix[:,dimen] > threshVal] = -1
	return retArray

#最佳单层决策树（即最小错误率的单层决策树）
def buildStump(dataArr,classLabels,D):
	dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
	m,n = shape(dataMatrix)

	#bestStump用于存储最佳决策树的相关信息
	numSteps = 10.0; bestStump = {}; bestClassEST = mat(zeros((m,1)))

	#minError首先初始化为无穷大
	minError = inf

	#在所有特征上遍历
	for i in range(n):
		rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
		stepSize = (rangeMax - rangeMin)/numSteps

		#在最小值和最大值之间根据步长遍历
		for j in range(-1,int(numSteps)+1):

			#在大于和小于之间切换不等式
			for inequal in ['lt','gt']:
				threshVal = (rangeMin + float(j) * stepSize)
				predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
				errArr = mat(ones((m,1)))
				errArr[predictedVals == labelMat] = 0
				weightedError = D.T*errArr
				#print ("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" %(i, threshVal, inequal, weightedError))
				if weightedError < minError:
					minError = weightedError
					bestClassEST = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump,minError,bestClassEST

#AdaBoost算法,针对错误的调节能力是其长处。DS代表单层决策树(decision stump)
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
	weakClassArr = []
	m = shape(dataArr)[0]

	#向量D包含了每个数据点的权重，开始都赋予了相等的值，在后续的迭代中会增加错分数据的权重，降低正确分类数据的权重。
	D = mat(ones((m,1))/m)

	#记录每个数据点的类别估计累计值
	aggClassEst = mat(zeros((m,1)))

	for i in range(numIt):

		#获取最小错误率的单层决策树，返回最小错误率以及估计的类别向量
		bestStump,error,classEst = buildStump(dataArr,classLabels,D)
		print("D:",D.T)

		#为了避免除0操作使用max(error,1e-16)
		alpha = float(0.5*log((1.0-error)/max(error,1e-16)))

		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)
		print("classEst",classEst.T)

		#计算下次迭代新的权重向量D
		expon = multiply(-1*alpha*mat(classLabels).T,classEst)
		D = multiply(D,exp(expon))
		D = D/D.sum()

		aggClassEst +=alpha*classEst
		print("aggClassEst:",aggClassEst.T)
		aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
		errorRate = aggErrors.sum()/m
		print("total error:",errorRate,"\n")
		if errorRate == 0.0: break
	return weakClassArr

if __name__ == "__main__":
	datMat,classLabels = loadSimpData()
	#print(datMat)
	#print(classLabels)
	D = mat(ones((5,1))/5)
	print(D)
	bestStump,minError,bestClassEST = buildStump(datMat,classLabels,D)
	print(bestStump,minError,bestClassEST)
	#exit()
	classifierArray = adaBoostTrainDS(datMat,classLabels,9)
	print(classifierArray)
