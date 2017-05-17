import sys
import time
from itertools import repeat
import io
import csv
import numpy as np
import DistanceMeasuresAnalyzer as dm
import math
import random
from statistics import mean
import highNGramAnalyzer as hn

def mergeD1ToD2 (d1,d2):

    for ng in d1:
        if d2.__contains__(ng)==False:
            d2[ng]={}
    for ng in d1:
        for t in d1[ng]:
            if d2[ng].__contains__(t):
                d2[ng][t]+=d1[ng][t]
            else:
                d2[ng][t]=d1[ng][t]
    return d2
def getAllNiGramsInEx (maxN=2):
    tempDic={}
    allNGramsInCorpus={}
    for i in range(40):
        trainSegs=hn.createTrainSetWithHighNGrams(i,maxN)
        testSegs=hn.createTestSetWithHighNGrams(i,maxN)

        for t in trainSegs:
            tempDic=hn.createNGramDistDic(t,maxN)
            mergeD1ToD2(tempDic,allNGramsInCorpus)
        for t in testSegs:
            tempDic=hn.createNGramDistDic(t,maxN)
            mergeD1ToD2(tempDic,allNGramsInCorpus)


    return allNGramsInCorpus

getAllNiGramsInEx(2)
