import sys
import time
from itertools import repeat
import io
import csv
import numpy as np
import math
import random
from statistics import mean
import DistanceMeasuresAnalyzer as dma
import highNGramAnalyzer as hn
import itertools

def getLabelsMatrix ():
    labels = []
    for j2 in range(41):
        with io.open('E:/challenge/training_data/user2SegLabels.csv', 'rt', encoding="utf8") as file:
            file.readline()
            labList = []
            for j in range(150):
                labList.append(file.readline().split(',')[j2])
            labels.append(labList)
    return labels

def createTrainSetForUser (i=0,maxN=50,splitSize=10):


    attributesLists = [['cosim' + str(k) for k in range(150)], ['pos' + str(k) for k in range(150)],
                       ['py' + str(k) for k in range(150)],
                       ['lev' + str(k) for k in range(150)], ['lcs' + str(k) for k in range(150)],['cosMini' + str(k) for k in range(150)],
                       ['jackSim' + str(k) for k in range(150)]]

    attrVec = list(itertools.chain(*attributesLists))
    attrVec.append('avgCosim')
    attrVec.append('label')

    attrVec = [s.replace("'", "") for s in attrVec]


    with io.open('E:/challenge/files/ds-train' + str(i) + '.csv', 'wt', encoding="utf8") as file1:
        file1.write(unicode(str(attrVec)[1:-1])+str("\n"))

    trainSegs = hn.createTrainSetWithHighNGrams(i, maxN)
    trainSegsTotal=[]
    for k in range(i,i+1):
        trainSegsTotal.extend(hn.createTrainSetWithHighNGrams(k,maxN))
        trainSegsTotal.extend(hn.createTestSetWithHighNGrams(k,maxN))
    cosimList=[]
    pyList=[]
    levList=[]
    lcsList=[]
    posList=[]
    cosMiniList=[]
    jackList=[]

    corpus = hn.createTrainCorpus(i, maxN)

    #foreach record
    for seg1 in trainSegs:
        distDic1 = hn.createNGramDistDic(seg1, maxN)
        fVec=[]
        cosimList = []
        pyList = []
        levList = []
        lcsList = []
        posList = []
        cosMiniList = []
        jackList = []

        for seg2 in trainSegsTotal:

            #cosinus similarity
            distDic2=hn.createNGramDistDic(seg2,maxN)
            sim=hn.calcCosimBetweenTwoComponents(corpus,distDic1,distDic2)
            cosimList.append(sim)

            #distance measures

            #py
            py=dma.pySim(seg1,seg2)
            pyList.append(py)

            #lcs

            lcsL = dma.lcs_length(seg1,seg2)
            lcs = dma.nLCS(seg1, seg2, lcsL)
            lcsList.append(lcs)

            #lev

            lev=dma.levenshtein(seg1,seg2)
            levList.append(1.0-lev/100.0)

            #pos

            pos=dma.calculatePosSimilarity(seg1,seg2)
            posList.append(pos)

            #cosMini sim

            cosMi=hn.calcSimPerMiniSeg(seg1,seg2,corpus,splitSize,maxN)
            cosMiniList.append(cosMi)

            #Jack-sim

            jackS=hn.calcJackardTwoSegs(distDic1,distDic2)
            jackList.append(jackS)


        #list2d = [[np.array(avgSim.values()).mean() for avgSim in cosimList],posList,pyList,levList,lcsList]
        list2d = [[np.array(avgSim.values()).mean() for avgSim in cosimList],posList,pyList,levList,lcsList,cosMiniList,jackList,[np.array([np.array(avgSim.values()).mean() for avgSim in cosimList]).mean()]]

        fVec= list(itertools.chain(*list2d))

        #get labels matrix
        labels=getLabelsMatrix()
        fVec.append(0)


        with io.open('E:/challenge/files/ds-train' + str(i) + '.csv', 'at', encoding="utf8") as file1:
            file1.write(unicode(str(fVec)[1:-1])+str("\n"))


def createTestSetForUser (i=0,maxN=50,splitSize=10):
    attributesLists = [['cosim' + str(k) for k in range(150)], ['pos' + str(k) for k in range(150)],
                       ['py' + str(k) for k in range(150)],
                       ['lev' + str(k) for k in range(150)], ['lcs' + str(k) for k in range(150)],
                       ['cosMini' + str(k) for k in range(150)],
                       ['jackSim' + str(k) for k in range(150)]]

    attrVec = list(itertools.chain(*attributesLists))
    attrVec.append("avgCosim")
    attrVec.append('label')

    attrVec = [s.replace("'", "") for s in attrVec]


    with io.open('E:/challenge/files/ds-test' + str(i) + '.csv', 'wt', encoding="utf8") as file1:
        file1.write(unicode(str(attrVec)[1:-1])+str("\n"))

    trainSegs=[]
    for k in range(i,i+1):
        trainSegs.extend(hn.createTrainSetWithHighNGrams(k, maxN))
        trainSegs.extend(hn.createTestSetWithHighNGrams(k,maxN))
    testSegs=hn.createTestSetWithHighNGrams(i,maxN)
    cosimList=[]
    pyList=[]
    levList=[]
    lcsList=[]
    posList=[]
    cosMiniList = []
    jackList = []

    corpus = hn.createTrainCorpus(i, maxN)

    #foreach record
    for seg1 in range(len(testSegs)):
        distDic1 = hn.createNGramDistDic(testSegs[seg1], maxN)
        fVec=[]
        cosimList = []
        pyList = []
        levList = []
        lcsList = []
        posList = []
        cosMiniList=[]
        jackList=[]


        for seg2 in trainSegs:

            #cosinus similarity
            distDic2=hn.createNGramDistDic(seg2,maxN)
            sim=hn.calcCosimBetweenTwoComponents(corpus,distDic1,distDic2)
            cosimList.append(sim)

            #distance measures

            #py
            py=dma.pySim(testSegs[seg1],seg2)
            pyList.append(py)

            #lcs

            lcsL = dma.lcs_length(testSegs[seg1],seg2)
            lcs = dma.nLCS(testSegs[seg1], seg2, lcsL)
            lcsList.append(lcs)

            #lev

            lev=dma.levenshtein(testSegs[seg1],seg2)
            levList.append(1.0-lev/100.0)

            #pos

            pos=dma.calculatePosSimilarity(testSegs[seg1],seg2)
            posList.append(pos)

            # cosMini sim

            cosMi = hn.calcSimPerMiniSeg(testSegs[seg1], seg2, corpus, splitSize, maxN)
            cosMiniList.append(cosMi)

            # Jack-sim

            jackS = hn.calcJackardTwoSegs(distDic1, distDic2)
            jackList.append(jackS)


            # list2d = [[np.array(avgSim.values()).mean() for avgSim in cosimList],posList,pyList,levList,lcsList]
        list2d = [[np.array(avgSim.values()).mean() for avgSim in cosimList], posList, pyList, levList, lcsList,
                  cosMiniList, jackList, [np.array([np.array(avgSim.values()).mean() for avgSim in cosimList]).mean()]]

        fVec= list(itertools.chain(*list2d))

        #get labels matrix
        labels=getLabelsMatrix()
        fVec.append(str(labels[i+1][50+seg1]))


        with io.open('E:/challenge/files/ds-test' + str(i) + '.csv', 'at', encoding="utf8") as file1:
            file1.write(unicode(str(fVec)[1:-1])+str("\n"))



for i in range(5,6):

    print("Started ds "+str(i))
    #createTrainSetForUser(i,10,25)
    createTestSetForUser(i,10,25)
    print("\n\nfinished ds "+str(i))



