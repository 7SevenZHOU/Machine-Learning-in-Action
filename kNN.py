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
plt.show(block=False)

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

plt.show()















