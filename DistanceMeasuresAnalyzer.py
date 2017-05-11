import sys
import time
from itertools import repeat
import io
import csv
import numpy as np
import math
from sets import Set
from difflib import SequenceMatcher
import highNGramAnalyzer as hn



def calculatePosSimilarity (s1,s2):
    count=0
    for t in range(len(s1)):
        if s1[t]==s2[t]:
            count+=1
    return float(float(count)/float(len(s1)))

def lcs_length(a, b):
    table = [[0] * (len(b) + 1) for _ in xrange(len(a) + 1)]
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if ca == cb else
                max(table[i][j - 1], table[i - 1][j]))
    return table[-1][-1]

def nLCS (a,b,lcsLength):
    return float(lcsLength/(math.sqrt(len(a)*len(b))))

def levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def pySim (a,b):
    sm = SequenceMatcher(None, a, b)
    pythonMatch = sm.ratio()
    return pythonMatch