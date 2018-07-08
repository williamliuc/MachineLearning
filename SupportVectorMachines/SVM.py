#-*- coding:utf-8 -*-
#辅助函数
from numpy import *
def loadDataSet(fileName):
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i 
    while (j==i):
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix=mat(dataMatIn);labelMat=mat(classLabels).transpose()
    b=0;m,n=shape(dataMatrix)
    alphas=mat(zeros((m,1)))
    #print(dataMatrix)
    #print(alphas.T)
    #exit()
    iter=0
    while(iter<maxIter):

        #alphaPairsChanged用于记录alpha是否已经进行优化
        alphaPairsChanged=0

        for i in range(m):

            #fxi为我们预测的类别
            fXi=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b

            #与真实label比对，计算误差Ei
            Ei=fXi - float(labelMat[i])

            #如果alpha可以更改（如果误差很大），进入优化过程
            if((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i]>0)):

                #随机选择第二个alpha
                j=selectJrand(i,m)

                #以第一个alpha值（alpha[i]）的误差计算方法来计算这个alpha值的误差。
                fXj=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                Ej=fXj-float(labelMat[j])

                #之所以使用copy函数而不是直接赋值，是因为python会通过引用的方式传递所有列表，所以必须明确的告知要分配新的内存，否则我们看不到新旧值的变化
                alphaIold=alphas[i].copy();
                alphaJold=alphas[j].copy();

                #保证alpha在0到C之间
                if(labelMat[i]!=labelMat[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])

                if L==H: print("L==H"); continue

                #eta是alpha[j]的最优修改量
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-\
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0: print ("eta>=0"); continue

                #计算新的alphas[j]，使用辅助函数对其进程调整
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta
                alphas[j]=clipAlpha(alphas[j],H,L)


                if (abs(alphas[j]-alphaJold)<0.00001): print ("j not moving enough"); continue

                #alphas[i]与alphas[j]同样的进行改变，虽然改变大小一样，但是方向相反（即如果一个增加，那么另外一个减少）
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])

                #对alpha[i]和alpha[j]进行优化之后，给这两个alpha值设置一个常数项b
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if(0<alphas[i])and(C>alphas[i]): b=b1
                elif(0<alphas[j])and(C>alphas[j]): b=b2
                else: b=(b1+b2)/2.0

                alphaPairsChanged+=1
                print ("iter: %d i:%d, pairs changed %d" %(iter,i,alphaPairsChanged)) 

        #iter变量存储的是在没有任何alpha改变的情况下遍历数据集的次数
        if(alphaPairsChanged==0): iter+=1
        else: iter=0
        print ("iteration number:%d" % iter)
        #exit()
    return b,alphas






if __name__=="__main__":
    dataArr,labelArr=loadDataSet('testSet.txt')
    #print(dataArr)
    #print(labelArr)
    #exit()
    b,alphas=smoSimple(dataArr,labelArr,0.6,0.001,40)
    print(b)

    #数组过滤，只对NumPy类型有用
    print(alphas[alphas>0])

    #支持向量个数
    print(shape(alphas[alphas>0]))

    #打印支持向量
    for i in range(100):
        if alphas[i]>0.0:print(dataArr[i],labelArr[i])
    
