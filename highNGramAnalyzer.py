import sys
import time
from itertools import repeat
import io
import csv
import numpy as np

import math
import random
from statistics import mean


def find_ngrams(input_list, n=50):
  return zip(*[input_list[i:] for i in range(n)])


def createNGramDistDic (input_list,maxN):

    allHighNGramsDic={}
    for i in range(1,maxN+1):
        distDic={}
        ngramsi=find_ngrams(input_list,i)
        for k in range(len(ngramsi)):
            if distDic.__contains__(ngramsi[k])==False:
                distDic[ngramsi[k]]=1
            else:
                distDic[ngramsi[k]]+=1

        allHighNGramsDic[i]=distDic
    return allHighNGramsDic



def createTrainSetWithHighNGrams (i=0,maxN=50):
    with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:
        trainSegments=[]
        seg=[]
        for j, row in enumerate(file.readlines()):
            if j >= 5000:
                continue
            seg.append(row[:-1])

            if (j+1)%100==0:
                trainSegments.append(seg)
                seg=[]
    return trainSegments
def createTrainCorpus (i=0,maxN=50):
    with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:
        trainTerms = {}

        ngramsTrain=createTrainSetWithHighNGrams(i,maxN)

        for seg in ngramsTrain:

            nDic=createNGramDistDic(seg,maxN)
            for ng in nDic:
                checkTerms={}
                for t in nDic[ng]:
                    if checkTerms.__contains__(t)==False:
                        checkTerms[t]=1
                    else:
                        checkTerms[t]+=1
                    if trainTerms.__contains__(t)==False:
                        trainTerms[t]=[1,1]
                    else:
                        trainTerms[t][0]=trainTerms[t][0]+nDic[ng][t]
                        if checkTerms[t]==1:
                            trainTerms[t][1]+=1

        return trainTerms


def createTestSetWithHighNGrams(i=0, maxN=50):
    with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:
        testSegments = []
        seg = []
        for j, row in enumerate(file.readlines()):
            if j < 5000:
                continue
            seg.append(row[:-1])

            if (j + 1) % 100 == 0:
                testSegments.append(seg)
                seg = []
    return testSegments


def findMaxFreqInSeg (seg):
    m=0.0
    for k in seg:
        if seg[k]>m:
            m=seg[k]
    return m

#each seg-param is a dic of ngrams dics, corpus is just a big dic with all mixed keys
def calcCosimBetweenTwoComponents(corpusDic, comp1Dic,comp2Dic):

    #calc tfidf and sum of tfidf in comp1
    tfidfDic1={}

    for ng in comp1Dic:
        maxFreqSeg1 = findMaxFreqInSeg(comp1Dic[ng])
        ngDic={}
        for t in comp1Dic[ng]:
            if corpusDic.__contains__(t):
                docsAppear = corpusDic[t][1]
            else:
                docsAppear=0.7
            ngDic[t] = (0.5 + 0.5 * float(comp1Dic[ng][t]) / (float(maxFreqSeg1) + 1)) * math.log(50.0 / (float(docsAppear) + 1))
        tfidfDic1[ng]=ngDic
    #remember this list starts from 0
    sumTfIdfPerNgComp1={el:0.0 for el in comp1Dic}
    for ng in comp1Dic:
        sumTfIdfPerNgComp1[ng]=sum(map(lambda x: x * x, tfidfDic1[ng].values()))

    # calc tfidf and sum of tfidf in comp2
    tfidfDic2 = {}

    for ng in comp2Dic:
        maxFreqSeg2 = findMaxFreqInSeg(comp2Dic[ng])
        ngDic = {}
        for t in comp2Dic[ng]:
            if corpusDic.__contains__(t):
                docsAppear = corpusDic[t][1]
            else:
                docsAppear=0.7
            ngDic[t] = (0.5 + 0.5 * float(comp2Dic[ng][t]) / (float(maxFreqSeg2) + 1)) * math.log(50.0 / (float(docsAppear) + 1))
        tfidfDic2[ng] = ngDic
    # remember this list starts from 0
    sumTfIdfPerNgComp2 = {el:0.0 for el in comp2Dic}
    for ng in comp2Dic:
        sumTfIdfPerNgComp2[ng]=sum(map(lambda x: x * x, tfidfDic1[ng].values()))

    cosimPerNgList = {}
    for ng in comp1Dic:
        numeratorSum=0.0
        for t1 in comp1Dic[ng]:
            if comp2Dic[ng].__contains__(t1):
                numeratorSum+=tfidfDic1[ng][t1]*tfidfDic2[ng][t1]

        sumTfIdfPerNgComp1[ng]=math.sqrt(sumTfIdfPerNgComp1[ng])
        sumTfIdfPerNgComp2[ng]=math.sqrt(sumTfIdfPerNgComp2[ng])

        cosimPerNgList[ng]=float(numeratorSum/(sumTfIdfPerNgComp1[ng]*sumTfIdfPerNgComp2[ng]))
    return cosimPerNgList


testSegsCur = createTestSetWithHighNGrams(5, 10)

for i in range(len(testSegsCur)):

    distDic1 = createNGramDistDic(testSegsCur[i], 10)

    for user in range(10):
        avgTotal = 0.0
        trainSegs = createTrainSetWithHighNGrams(user, 10)
        testSegs = createTestSetWithHighNGrams(user, 10)
        corpus = createTrainCorpus(user, 10)

        sVsTrainAvgScore=0.0
        for j in range(50):


            distDic2=createNGramDistDic(trainSegs[j],10)
            cosimList=calcCosimBetweenTwoComponents(corpus,distDic1,distDic2)
            cosimListAvg=np.array(cosimList.values()).mean()
            sVsTrainAvgScore+=cosimListAvg
        sVsTrainAvgScore/=50.0

        sVsTestAvgScore = 0.0
        for j in range(100):
            distDic2 = createNGramDistDic(testSegs[j], 10)
            cosimList = calcCosimBetweenTwoComponents(corpus, distDic1, distDic2)
            cosimListAvg = np.array(cosimList.values()).mean()
            sVsTestAvgScore += cosimListAvg
        sVsTestAvgScore /= 100.0
        avgTotal+=(sVsTrainAvgScore*0.33333+sVsTestAvgScore*0.666)

    avgTotal/=10

    print(str(i)+" is: "+str(avgTotal))




print("finished high NGram")