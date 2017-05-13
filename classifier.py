import numpy as np
import io
import dA as da
import datasetGen as ds
"""
data_train = np.loadtxt('E:/challenge/files/ds-train5.csv', delimiter=',')
X = data_train[:, 1:]
y = data_train[:, 0].astype(np.int)
clf = ExtraTreesClassifier(n_estimators=100).fit(X, y)


data_test = np.loadtxt('E:/challenge/files/ds-test5.csv', delimiter=',')
print(clf.predict(data_test))
"""

def trainAndTestDA(i=5,maxN=50):
    dA = da.dA(n_visible=250, n_hidden=83)
    print ("train starts")
    for iter in range(100):
        with io.open('E:/challenge/files/ds-train' + str(i) + '.csv', 'rt', encoding="utf8") as file:
            file.readline()

            for j in range(50):
                fields=file.readline().split(',')[:-1]
                fields=[float(x) for x in fields]
                score=dA.train(input=np.array(fields))
                print(score)
    print ("test start")
    with io.open('E:/challenge/files/ds-test' + str(i) + '.csv', 'rt', encoding="utf8") as file:
        file.readline()
        for j in range(100):
            fields=file.readline().split(',')[:-1]
            fields=[float(x) for x in fields]
            score=dA.feedForward(input=np.array(fields))
            print(str(j)+" :"+str(score))



trainAndTestDA(5,10)