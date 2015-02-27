import math
import operator
import numpy as np
import matplotlib.pyplot as plt
import csv

'''
README

I was unable to get an arff reader to work on either my mac or at the CS dept computers. Therefore I uploaded the data
as a csv and hardcoded the np arrays after reading them in.

To calculate EucDist between two points, use:
EucDist(a,b,length) where length specfies how many numerical variables there are
to calculate distance, assuming all non-numeric variables exist after the numbers.
eg for a= [ 6.5,  3. ,  5.8,  2.2], and b = [ 5.1,  3.3,  1.7,  0.5]
EucDist(a,b,4)

To find the K-NN of a point, enter in the training set, the test observation, and k to test using:
NearestNeighbors(train, testobs, k)

To Predict the class of an item using the test observation and the list of Nearest neighbors use:
PredictClass(test,neighbors)


To retrieve the results for this assignment enter either:
results
or
addColsk(test,train)

To perform the same analysis on a different dataset, enter in the training array 'train' and
the testing array test into addColsk(test,train). Or for a single obs:
enter
newobs = np.array([[ 6.7,  3.1,  4.4,  1.4]])
and then test using the same training set:
addColsk(newobs, train)

returning: array([['6.7', '3.1', '4.4', '1.4', 'x0', 'versicolor', 'versicolor',
                'versicolor', 'versicolor', 'versicolor']], 
                  dtype='|S32')
'''



train = np.array([(6.1, 2.9, 4.7, 1.4, b'versicolor'),
        (6.0, 2.7, 5.1, 1.6, b'versicolor'),
        (5.4, 3.0, 4.5, 1.5, b'versicolor'),
        (5.6, 2.5, 3.9, 1.1, b'versicolor'),
        (4.9, 3.1, 1.5, 0.1, b'setosa'),  (6.3, 3.3, 6.0, 2.5, b'setosa'),
        (6.9, 3.1, 5.4, 2.1, b'virginica'),
        (4.9, 2.5, 4.5, 1.7, b'virginica'),
        (6.4, 3.2, 5.3, 2.3, b'virginica'),
        (7.4, 2.8, 6.1, 1.9, b'virginica'),  (6.2, 2.8, 4.8, 1.8, b'setosa'),
        (7.2, 3.0, 5.8, 1.6, b'virginica'),
        (6.4, 3.1, 5.5, 1.8, b'virginica'),
        (7.9, 3.8, 6.4, 2.0, b'virginica'),
        (6.7, 3.3, 5.7, 2.1, b'virginica'),
        (6.2, 2.9, 4.3, 1.3, b'versicolor'),
        (5.6, 2.9, 3.6, 1.3, b'versicolor'),
        (6.3, 2.5, 4.9, 1.5, b'versicolor'),
        (6.5, 2.8, 4.6, 1.5, b'versicolor'),
        (5.8, 2.7, 4.1, 1.0, b'versicolor'),
        (4.6, 3.2, 1.4, 0.2, b'virginica'),
        (6.2, 3.4, 5.4, 2.3, b'virginica'),  (4.8, 3.4, 1.9, 0.2, b'setosa'),
        (5.1, 3.8, 1.9, 0.4, b'setosa'),
        (6.3, 2.3, 4.4, 1.3, b'versicolor'),
        (6.0, 2.2, 5.0, 1.5, b'virginica'),  (5.0, 3.2, 1.2, 0.2, b'setosa'),
        (6.0, 3.4, 4.5, 1.6, b'versicolor'),
        (6.6, 3.0, 4.4, 1.4, b'versicolor'),
        (5.0, 3.0, 1.6, 0.2, b'setosa'),
        (5.6, 2.7, 4.2, 1.3, b'versicolor'),
        (5.4, 3.4, 1.5, 0.4, b'setosa'),
        (5.5, 2.6, 4.4, 1.2, b'versicolor'),
        (4.6, 3.4, 1.4, 0.3, b'setosa'),  (4.5, 2.3, 1.3, 0.3, b'setosa'),
        (6.4, 3.2, 4.5, 1.5, b'virginica'),  (5.4, 3.9, 1.3, 0.4, b'setosa'),
        (5.1, 2.5, 3.0, 1.1, b'versicolor'),
        (5.9, 3.0, 4.2, 1.5, b'versicolor'),
        (4.8, 3.4, 1.6, 0.2, b'setosa'),
        (5.0, 2.0, 3.5, 1.0, b'versicolor'),
        (5.6, 2.8, 4.9, 2.0, b'virginica'),  (4.4, 2.9, 1.4, 0.2, b'setosa'),
        (6.4, 2.8, 5.6, 2.1, b'virginica'),  (4.6, 3.1, 1.5, 0.2, b'setosa'),
        (6.3, 2.5, 5.0, 1.9, b'virginica'),
        (5.7, 2.8, 4.5, 1.3, b'versicolor'),
        (5.8, 2.7, 5.1, 1.9, b'virginica'),  (4.4, 3.0, 1.3, 0.2, b'setosa'),
        (5.2, 4.1, 1.5, 0.1, b'setosa'),  (5.0, 3.4, 1.5, 0.2, b'setosa'),
        (5.1, 3.7, 1.5, 0.4, b'setosa'),  (5.0, 3.5, 1.3, 0.3, b'setosa'),
        (6.4, 2.8, 5.6, 2.2, b'virginica'),  (4.9, 3.0, 1.4, 0.2, b'setosa'),
        (5.0, 3.6, 1.4, 0.2, b'setosa'),
        (5.0, 2.3, 3.3, 1.0, b'versicolor'),
        (7.7, 3.8, 6.7, 2.2, b'virginica'),
        (6.0, 2.9, 4.5, 1.5, b'versicolor'),
        (4.9, 3.1, 1.5, 0.1, b'setosa'),
        (5.7, 2.6, 3.5, 1.0, b'versicolor'),
        (4.9, 2.4, 3.3, 1.0, b'versicolor'),
        (5.8, 2.8, 5.1, 2.4, b'virginica'),
        (5.5, 2.4, 3.7, 1.0, b'versicolor'),
        (6.7, 3.1, 5.6, 2.4, b'virginica'),
        (5.9, 3.0, 5.1, 1.8, b'versicolor'),
        (6.3, 3.4, 5.6, 2.4, b'virginica'),
        (6.3, 2.7, 4.9, 1.8, b'virginica'),
         (5.5, 4.2, 1.4, 0.2, b'setosa'),
        (6.8, 3.0, 5.5, 2.1, b'virginica'),
        (5.8, 2.6, 4.0, 1.2, b'versicolor'),
        (4.7, 3.2, 1.3, 0.2, b'setosa'),
         (6.7, 3.0, 5.2, 2.3, b'virginica'),
        (5.1, 3.8, 1.5, 0.3, b'setosa'),
         (4.8, 3.0, 1.4, 0.3, b'setosa')], dtype= ([('x0', '<f8'), ('x1', '<f8'), ('x2', '<f8'), ('x3', '<f8'), ('y', 'S10')]))

trainx = np.transpose(np.vstack([train['x0'],train['x1'],train['x2'],train['x3']]))
trainy = train['y'].reshape(75,1)

test = np.array([[ 6.7,  3.1,  4.4,  1.4],
       [ 4.4,  3.2,  1.3,  0.2],
       [ 5.3,  3.7,  1.5,  0.2],
       [ 7.7,  2.8,  6.7,  2. ],
       [ 5.1,  3.5,  1.4,  0.2],
       [ 6.5,  3. ,  5.2,  2. ],
       [ 7.1,  3. ,  5.9,  2.1],
       [ 6.4,  2.7,  5.3,  1.9],
       [ 5.2,  2.7,  3.9,  1.4],
       [ 7. ,  3.2,  4.7,  1.4],
       [ 7.2,  3.2,  6. ,  1.8],
       [ 5.4,  3.7,  1.5,  0.2],
       [ 5.6,  3. ,  4.5,  1.5],
       [ 5.9,  3.2,  4.8,  1.8],
       [ 5.1,  3.4,  1.5,  0.2],
       [ 6.9,  3.1,  4.9,  1.5],
       [ 6. ,  2.2,  4. ,  1. ],
       [ 4.7,  3.2,  1.6,  0.2],
       [ 4.6,  3.6,  1. ,  0.2],
       [ 5.6,  3. ,  4.1,  1.3],
       [ 5.5,  3.5,  1.3,  0.2],
       [ 5.5,  2.4,  3.8,  1.1],
       [ 5.1,  3.8,  1.6,  0.2],
       [ 6.3,  3.3,  4.7,  1.6],
       [ 6.6,  2.9,  4.6,  1.3],
       [ 7.7,  3. ,  6.1,  2.3],
       [ 6.4,  2.9,  4.3,  1.3],
       [ 6.9,  3.1,  5.1,  2.3],
       [ 6.7,  3. ,  5. ,  1.7],
       [ 4.3,  3. ,  1.1,  0.1],
       [ 7.7,  2.6,  6.9,  2.3],
       [ 6.7,  3.3,  5.7,  2.5],
       [ 6.7,  2.5,  5.8,  1.8],
       [ 1. ,  3.1,  1.6,  0.2],
       [ 5.7,  4.4,  1.5,  0.4],
       [ 6.5,  3. ,  5.5,  1.8],
       [ 6.1,  3. ,  4.9,  1.8],
       [ 5.4,  3.4,  1.7,  0.2],
       [ 6.5,  3.2,  5.1,  2. ],
       [ 5.2,  3.4,  1.4,  0.2],
       [ 5.7,  3. ,  4.2,  1.2],
       [ 5.5,  2.3,  4. ,  1.3],
       [ 5. ,  3.4,  1.6,  0.4],
       [ 5.8,  2.7,  5.1,  1.9],
       [ 6.1,  2.8,  4. ,  1.3],
       [ 5.7,  2.5,  5. ,  2. ],
       [ 6.3,  2.9,  5.6,  1.8],
       [ 4.9,  3.1,  1.5,  0.1],
       [ 6.8,  3.2,  5.9,  2.3],
       [ 6.9,  3.2,  5.7,  2.3],
       [ 6.7,  3.1,  4.7,  1.5],
       [ 5.7,  2.8,  4.1,  1.3],
       [ 5. ,  3.5,  1.6,  0.6],
       [ 5.4,  3.9,  1.7,  0.4],
       [ 5.2,  3.5,  1.5,  0.2],
       [ 6.1,  2.8,  4.7,  1.2],
       [ 5.7,  2.9,  4.2,  1.3],
       [ 5.8,  2.7,  3.9,  1.2],
       [ 5. ,  3.3,  1.4,  0.2],
       [ 6.8,  2.8,  4.8,  1.4],
       [ 6.3,  2.8,  5.1,  1.5],
       [ 6.2,  2.2,  4.5,  1.5],
       [ 6. ,  3. ,  4.8,  1.8],
       [ 5.1,  3.5,  1.4,  0.3],
       [ 5.7,  3.8,  1.7,  0.3],
       [ 6.1,  3. ,  4.6,  1.4],
       [ 5.8,  4. ,  1.2,  0.2],
       [ 7.2,  3.6,  6.1,  2.5],
       [ 6.1,  2.6,  5.6,  1.4],
       [ 5.5,  2.5,  4. ,  1.3],
       [ 7.3,  2.9,  6.3,  1.8],
       [ 4.8,  3. ,  1.4,  0.1],
       [ 7.6,  3. ,  6.6,  2.1],
       [ 6.5,  3. ,  5.8,  2.2],
       [ 5.1,  3.3,  1.7,  0.5]])



def EucDist(a, b, length):
    distance = 0
    for x in range(length):
        distance += pow((a[x] - b[x]), 2)
    return math.sqrt(distance)

def NearestNeighbors(train, testobs, k):
  #returns list of k closest points and distances to test
    distances = []
    length = len(testobs)-1
    for x in range(len(train)):
        dist = EucDist(testobs, train[x], length)
        distances.append((train[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def PredictClass(test, neighbors):
   #takes in list of k closest points to test and test point
    votes = {}
            #Unweighted votes and total distance
    length = len(test)
    if len(neighbors) == 1:
        return neighbors[0][4]
    else:
        for x in range(len(neighbors)):
            Type = neighbors[x][-1]
            if Type in votes:
#not recognizing Type in votes
                votes[str(Type)][0] += 1    #total votes
                votes[str(Type)][1] += EucDist(test, neighbors[x], length)   #total distance
            else:
                votes[str(Type)] = [1,0]
                votes[str(Type)][1] = EucDist(test, neighbors[x], length)
        finalVotes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
                        #if tied choose minimum total distance
        if len(finalVotes) == 1:
            return finalVotes[0][0]
        elif finalVotes[0][1][0] == finalVotes[1][1][0]:
            values = [finalVotes[0][1][1], finalVotes[1][1][1]]
            return finalVotes[values.index(min(values))][0]
        else:
            return finalVotes[0][0]

def addColsk(testset, trainset):    #k 1,3,5,7,9
    length = len(testset)
    colx = np.empty((testset.shape[0],1), dtype='S12')
    col1 = np.empty((testset.shape[0],1), dtype='S12')
    col3 = np.empty((testset.shape[0],1), dtype='S12')
    col5 = np.empty((testset.shape[0],1), dtype='S12')
    col7 = np.empty((testset.shape[0],1), dtype='S12')
    col9 = np.empty((testset.shape[0],1), dtype='S12')
    for x in range(length):
        colx[x,0] = 'x' + str(x)
        col1[x,0] = PredictClass(testset[x], NearestNeighbors(trainset, testset[x], 1))
        col3[x,0] = PredictClass(testset[x], NearestNeighbors(trainset, testset[x], 3))
        col5[x,0] = PredictClass(testset[x], NearestNeighbors(trainset, testset[x], 5))
        col7[x,0] = PredictClass(testset[x], NearestNeighbors(trainset, testset[x], 7))
        col9[x,0] = PredictClass(testset[x], NearestNeighbors(trainset, testset[x], 9))
        
    return np.hstack((test, colx, col1,col3,col5,col7,col9))


results = addColsk(test,train)

