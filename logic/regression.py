#encoding:utf-8
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))-1
    dataMat = []; labelMat = []
    fr = open(fileName)
    #print fr.readlines()
    for line in fr.readlines():
        lineArr=[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#标准线性回归
def standRegres(xArr,yArr):
    xMat = mat(xArr);yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print "this matrix is singular,cannot do inverse"
        return
    ws = xTx.I*(xMat.T*yMat)
#绘图
    fig,ax=plt.subplots()
    xcopy=xMat.copy()
    xcopy.sort(0)
    yhat=xcopy*ws
    ax.plot(xcopy[:,1],yhat)
    ax.scatter(xMat[:,1],yMat) 
    plt.show()
#使用numpy库中的corrcoef函数计算预测值与真实值的相关系数
    print corrcoef(yhat.T,yMat.T)
    return ws

#局部加权线性回归
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):	#gao si he,lei si yu knn suan fa
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T*(weights*xMat)
    if linalg.det(xTx) == 0.0:
        print "this matrix is singular, cannot do inverse"
        return
    ws = xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    xMat=mat(xArr)
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
#绘图
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    fig,ax = plt.subplots()
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1],mat(yArr).T)
    plt.show()
    return yHat

    
if __name__=="__main__":

#标准线性回归测试
#    datamat,labelmat=loadDataSet('ex0.txt')
 #   ws = standRegres(datamat,labelmat)
 #   print ws


#局部加权线性回归测试
    datamat,labelmat=loadDataSet('ex0.txt')
    lwlrTest(datamat,datamat,labelmat,0.1)
    #lwlr(datamat[1],datamat,labelmat,0.1)
    #lwlr(datamat[150],datamat,labelmat,0.1)
