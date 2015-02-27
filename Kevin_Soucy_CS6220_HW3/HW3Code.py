import math
import operator
import numpy as np
import csv

'''
README

I was unable to get an arff reader to work on either my mac or at the CS dept computers.
So I created csvs using Weka. 

To calculate EucDist between two points, use:
EucDist(a,b,length) where length specfies how many numerical variables there are
to calculate distance, assuming all non-numeric variables exist after the numbers.
eg for a= [ 6.5,  3. ,  5.8,  2.2], and b = [ 5.1,  3.3,  1.7,  0.5]
EucDist(a,b,4)

'''

#Set Path to segment.csv
def segmentcsvpath():
    print ("Please set path to folder containing segment.csv eg '\\Users\\soucy.ke\\Desktop\\segment.csv' please enter '\\\\Users\\\\soucy.ke\\\\Desktop\\\\'  below with quotations: ")
    global segmentpath
    segmentpath = input()
    print('Thank you. Please wait...')
print(segmentcsvpath())

#Mac segmentpath = '/Users/Kevin/Desktop/'
#Windows Enter segmentpath = '\\\\Users\\\\soucy.ke\\\\Desktop\\\\'

Indices = [775, 1020, 200, 127, 329, 1626, 1515, 651, 658, 328, 1160, 108, 422, 88, 105, 261, 212, 1941, 1724, 704, 1469, 635, 867, 1187, 445, 222, 1283, 1288, 1766, 1168, 566, 1812, 214, 53, 423, 50, 705, 1284, 1356, 996, 1084, 1956, 254, 711, 1997, 1378, 827, 1875, 424, 1790, 633, 208, 1670, 1517, 1902, 1476, 1716, 1709, 264, 1, 371, 758, 332, 542, 672, 483, 65, 92, 400, 1079, 1281, 145, 1410, 664, 155, 166, 1900, 1134, 1462, 954, 1818, 1679, 832, 1627, 1760, 1330, 913, 234, 1635, 1078, 640, 833, 392, 1425, 610, 1353, 1772, 908, 1964, 1260, 784, 520, 1363, 544, 426, 1146, 987, 612, 1685, 1121, 1740, 287, 1383, 1923, 1665, 19, 1239, 251, 309, 245, 384, 1306, 786, 1814, 7, 1203, 1068, 1493, 859, 233, 1846, 1119, 469, 1869, 609, 385, 1182, 1949, 1622, 719, 643, 1692, 1389, 120, 1034, 805, 266, 339, 826, 530, 1173, 802, 1495, 504, 1241, 427, 1555, 1597, 692, 178, 774, 1623, 1641, 661, 1242, 1757, 553, 1377, 1419, 306, 1838, 211, 356, 541, 1455, 741, 583, 1464, 209, 1615, 475, 1903, 555, 1046, 379, 1938, 417, 1747, 342, 1148, 1697, 1785, 298, 1485, 945, 1097, 207, 857, 1758, 1390, 172, 587, 455, 1690, 1277, 345, 1166, 1367, 1858, 1427, 1434, 953, 1992, 1140, 137, 64, 1448, 991, 1312, 1628, 167, 1042, 1887, 1825, 249, 240, 524, 1098, 311, 337, 220, 1913, 727, 1659, 1321, 130, 1904, 561, 1270, 1250, 613, 152, 1440, 473, 1834, 1387, 1656, 1028, 1106, 829, 1591, 1699, 1674, 947, 77, 468, 997, 611, 1776, 123, 979, 1471, 1300, 1007, 1443, 164, 1881, 1935, 280, 442, 1588, 1033, 79, 1686, 854, 257, 1460, 1380, 495, 1701, 1611, 804, 1609, 975, 1181, 582, 816, 1770, 663, 737, 1810, 523, 1243, 944, 1959, 78, 675, 135, 1381, 1472]
datax = np.loadtxt(segmentpath + 'segmentx.csv', delimiter=',', skiprows=1)
datay = np.genfromtxt(segmentpath + 'segmenty.csv', delimiter=',', skiprows=1)
datay.shape = (len(datay),1)


def EucDist(a, b):
    distance = 0
    for x in range(len(a)):
        distance += pow((a[x] - b[x]), 2)
    return math.sqrt(distance)

def normalize(col): #col -> norm col
    m = np.mean(col)
    s = np.std(col)
    col2 = np.zeros((col.shape))
    if s == 0:
        for x in range(len(col2)):
            col2[x] = m
    else:
        for x in range(len(col2)):
            col2[x] = (col[x]-m) / s
    return col2

def norm(arr): # arr -> each col
    arr2 = np.zeros(arr.shape)
    for x in range(arr.shape[1]):
        arr2[:,x] = normalize(arr[:,x])
    return arr2



                            
'''
#terminate when cluster vectors are equal from one step to the next or you iterate 50 times
#Centroids is a list of indices corresponding to rows in data. Then it becomes the averaage of all instances in that cluster
            
#Algo Iterate 50 times, cluster matrix, data matrix with added col of cluster indices.            

'''

# centroids: array of shape = (k by 19) equal to mean points of each cluster. 
# Clusters is dictionary of k lists of indices of each row belonging to this cluster.                   

def calcCentroids(data, lastcentroids, clusters):
    centroids = np.zeros((len(clusters),data.shape[1]))
    for i in clusters:
        centroids[i] = np.mean(data[clusters[i]], axis=0)
    return centroids

def assigncluster(data, centroids): #for every row in data, calculate dist to each centroid, return closest centroid, 
    clusters = {}
    for x in range(len(data)):
        mindistx = [-1, float("inf")]           #['clus', 'dist']
        for i in range(len(centroids)):
            disti = EucDist(data[x,], centroids[i,])
            if mindistx[1] > disti:
                mindistx = [i, disti]
            else:
                pass                                                                  
        try:
            clusters[mindistx[0]].append(x)
        except KeyError:
            clusters[mindistx[0]] = [x]
    return clusters

def calcSSE(data, kmeans): #data, centroids and clusters
    centroids = kmeans[0]
    clusters = kmeans[1]
    totalsum = [] #list of errors for each cluster
    for m in range(len(centroids)): #for each centroid
        clussum = [] #list of errors for each obs
        for x in clusters[m]:  #refers to keys? #calc dist for each pt in cluster to centroid
            clussum.append(EucDist(data[x,],centroids[m])**2)   #?? refer to data[x,] row  from clusters
        totalsum.append(np.sum(clussum))
    return np.sum(totalsum)

    
#append i to clusters dict with key corresponding to position of centroid in centroids 
#compute closest centroid
                        
#for i in range(len(centroids)):
 #          closestcentroid = min([(i[0], EucDist(x, centroids[i]

#returns dictionary of k lists of indices of each row belonging to this cluster.  

def kmeans(data, k, InitCentroids, maxiters=50):    #returns list of k closest points and distances to test   12*25 = 300, so iterate
    lastcentroids = np.zeros(data[InitCentroids,].shape)         # InitCentroids indexed by Indices
    centroids = data[InitCentroids,]
    lastclusters = {1:1}
    clusters = {}
    # Calculate  for each cluster to get newcentroids
    while ((clusters == lastclusters) == False) and (maxiters >0):
            lastcentroids = centroids
            lastclusters = clusters
            clusters = assigncluster(data, centroids)
            centroids = calcCentroids(data, lastcentroids, clusters)
            maxiters -= 1
    return(centroids, clusters)


def repeatKmeans(data,  k, Indices = Indices, reps = 25):  #Computes Average SSE
    results = np.zeros((reps, 1)) 
    for r in range(reps):
        InitCentroids = Indices[(r*k):((r*k)+k)]    
        results[r] = calcSSE(data, kmeans(data, k, InitCentroids))
    return results
    #return(np.mean(results), np.std(results))

