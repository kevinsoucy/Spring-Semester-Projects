README Python 

1. Run .py in Idle interpreter
2. Set File Path
1. Please set path to folder containing segment.csv 
1. For Windows eg '\Users\soucy.ke\Desktop\segment.csv' please enter '\\Users\\soucy.ke\\Desktop\\'  below with quotations: 

3. Functions:  MAKE SURE TO REMOVE NON-NUMERICAL COLUMNS FROM DATA PRIOR TO USING THESE FUNCTIONS

1. EucDist(a, b)
2. normalize(col)		normalizes column or row vector
3. norm(arr)		normalizes array
4. kmeans(data, k, InitCentroids, maxiters=50)		
1. returns dictionary of clusters, eg {1: [0,2,3], where the dictkeys are the cluster numbers and dict.items are lists of the index row values from the segments dataset. 
2. Also returns Centroids: the mean points for each cluster. 
5. repeatKmeans(data,  k, Indices = Indices, reps = 25): 
1. returns array of the SSE from each repitition.
