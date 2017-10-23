from numpy import *
import operator

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

group,labels=createDataSet()
#print(group)
#print(labels)

def MykNN(inX,dataset,labels,k):
    datasetSize=dataset.shape[0]
    diffMat=tile(inX,(datasetSize,1))-dataset
    sqdiffMat=diffMat**2
    sqDistances=sqdiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1

    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

result=MykNN([0,0],group,labels,3)
print(result)


def file2matrix(filename):
    labelsToNumber={'largeDoses':1,'smallDoses':2,'didntLike':3}
    fr=open(filename)
    numberOfLines=len(fr.readlines())
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    fr=open(filename)
    index=0

    for line in fr.readlines():
        line=line.strip()
        listFromLine=line.split('\t')
        #mark!
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(labelsToNumber[listFromLine[-1]]))
        index +=1
    return returnMat,classLabelVector

datingDataMat,datingLabels=file2matrix('datingTestSet.txt')

print(datingDataMat)
print(datingLabels)

import matplotlib
import matplotlib.pyplot as plt

fig=plt.figure()
ax=fig.add_subplot(111)
#mark! scatter(x,y,scale,color)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15*array(datingLabels),15*array(datingLabels))
plt.show()

def autoNorm(dataset):
    #The 0 in dataSet.min(0) allows you to take the minimums from the columns, not the rows.
    minVals=dataset.min(0)
    maxVals=dataset.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataset))
    m=dataset.shape[0]
    #Element-wise division
    normDataSet=dataset-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

nortMat,ranges,minVals=autoNorm(datingDataMat)

print(nortMat)
print(ranges)
print(minVals)




def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('datingTestSet.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=MykNN(normMat[i,:],normMat[numTestVecs:m,:],\
            datingLabels[numTestVecs:m],3)
        print('the classifier came back with: %d,the real answer is %d'%(classifierResult,datingLabels[i]))
        if(classifierResult!=datingLabels[i]):
            errorCount+=1.0
    print('the total error rate is:%f'%(errorCount/float(numTestVecs)))

datingClassTest()

def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(input('percentage of time spent playing video games?'))
    ffMiles=float(input('requent flier miles earned per year?'))
    iceCream=float(input('liters of ice cream sonsumed per year?'))
    datingDataMat,datingLabels=file2matrix('datingTestSet.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=MykNN((inArr-minVals)/ranges,normMat,datingLabels,3)
    print('You will probably like this person:',resultList[classifierResult-1])

#classifyPerson()

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

import os

def handwritingClassTest():
    hwLabels=[]
    trainingFileList=os.listdir('trainingDigits')
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('trainingDigits/%s'%fileNameStr)
    testFileList=os.listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/%s'%fileNameStr)
        classifierResult=MykNN(vectorUnderTest,trainingMat,hwLabels,3)
        print('the classifier came back with:%d,the real answer is:%d'%(classifierResult,classNumStr))
        if(classifierResult!=classNumStr):
            errorCount+=1.0
    print('\nthe total number of errors is %d'%errorCount)
    print('\nthe total error rate is:%f'%(errorCount/float(mTest)))

handwritingClassTest()






















