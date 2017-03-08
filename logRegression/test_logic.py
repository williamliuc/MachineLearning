from numpy import *
def loaddata():
    train_x=[]
    train_y=[]
    fileIn=open('C:/Users/lc/.spyder/logic/test.txt')
    for line in fileIn.readlines():
        alow=line.strip().split()#split()函数默认从空格处分开
        train_x.append([alow[0],alow[1]])
        train_y.append(alow[2])
    return mat(train_x),mat(train_y).transpose()

print "load data..."

train_x,train_y=loaddata()
print train_x
numSamples, numFeatures = train_x.shape
print numSamples,numFeatures
weights = ones((numFeatures, 1))
print weights
output=1/(1-exp(-train_x * weights))
print output