from math import log

def calcShannonEntropy(dataSet):
    numEntry=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEntropy=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntry
        shannonEntropy-=prob*log(prob,2)
    return shannonEntropy


def createDataSet():
    dataSet=[[1,1,'yes'],\
    [1,1,'yes'],\
    [1,0,'no'],\
    [0,1,'no'],\
    [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

myData,labels=createDataSet()
print(myData)

print(calcShannonEntropy(myData))

#myData[0][-1]='maybe'
#print(calcShannonEntropy(myData))

def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

print(splitDataSet(myData,0,1))
print(splitDataSet(myData,0,0))

def chooseBestFeatrueToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEntropy(dataset)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(data,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+= prob*calcShannonEntropy(subDataSet)
        infoGain=baseEntropy-newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature






















