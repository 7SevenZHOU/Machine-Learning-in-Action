'''
Naive Bayes: naive means we can asuume all the features are independent and equally important.
'''

def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],\
                ['maybe','not','take','him','to','dog','park','stupid'],\
                ['my','dalmation','is','so','cute','I','love','him'],\
                ['stop','posting','stupid','worthless','garbage'],\
                ['mr','licks','ate','my','steak','how','to','stop','him'],\
                ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1] #1 is abusive
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        # | operator means set union
        vocabSet=vocabSet|set(document)
    reutrn list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:print('the word:%s is not in my Vocabulary!'%word)
    return returnVec





