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
    featuresName=['no surfacing','flippers']
    return dataSet,featuresName

myData,featuresNameList=createDataSet()
print(myData)
print(featuresNameList)

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
    baseEntropy=calcShannonEntropy(dataSet)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        #mark!
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+= prob*calcShannonEntropy(subDataSet)
        infoGain=baseEntropy-newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

print(chooseBestFeatrueToSplit(myData))

import operator

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classList.keys():
            classList[vote]=0
        classList[vote]+=1
    sortedClassCount=sorted(classCount.item(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,featuresNameList):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeature=chooseBestFeatrueToSplit(dataSet)
    bestFeatureName=featuresNameList[bestFeature]
    myTree={bestFeatureName:{}}
    del(featuresNameList[bestFeature])
    featValues=[example[bestFeature] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subFeaturesNameList=featuresNameList[:]
        myTree[bestFeatureName][value]=createTree(splitDataSet(dataSet,bestFeature,value),subFeaturesNameList)
    return myTree

myTree=createTree(myData,featuresNameList)
print(myTree)


def classify(inputTree,featLabels,testVec):
    firstStr=list(inputTree)[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw=open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr=open(filename,'rb')
    return pickle.load(fr)

#storeTree(myTree,'classifierStorage.txt')
#print(grabTree('classifierStorage.txt'))

fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesFeatNames=['age','prescript','astigmatic','tearRate']
lensesTree=createTree(lenses,lensesFeatNames)
print(lensesTree)
import treePlotter
treePlotter.createPlot(lensesTree)









































