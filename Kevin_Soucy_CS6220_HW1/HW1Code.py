import numpy as np
import matplotlib.pyplot as plt

'''
Change path to folder containing hw1data folder
    eg if hw1-data folder contains train-1000-100.csv at '/Users/Kevin/Desktop/hw1-data/train-1000-100.csv'
    Enter: /Users/Kevin/Desktop/     (without quotations)
    '''

#set hw1datapath
def hw1dataFolderpath():
    print ("Please set path to folder containing hw1data folder eg if hw1data folder contains train-1000-100.csv at '\\Users\\soucy.ke\\Desktop\\hw1data\\train-1000-100.csv' please enter '\\\\Users\\\\soucy.ke\\\\Desktop\\\\'  below with quotations: ")
    global hw1datapath
    hw1datapath = input()
    print('Thank you. Please wait...')
print(hw1dataFolderpath())

"\\Users\\soucy.ke\\Desktop\\"

'''
Readme
pip install numpy, matplotlib

Key Functions
Q1
addCol1(matrix)  adds column of ones at beginning of data matrix, for intercept values
calcMSE(theta, X, y)
L2LR(X, y, alpha): runs L2 regularized linear regression
traintest(train, test, alphas): #Creates matrix of alphas and testing MSEs for range of alphas, eg alphas = range(0,151)
plotmse(mod) : takes in matrix from traintest() function and plots alphas(x) vs. MSE 

Q2
plotLCs() plots learning curves for alphas 1,25,150, LCs are averaged after 10 trials each.

Q3
FindBestLambda(k, trainingset, maxalpha,justmin=1)
# Set justmin=0 to get table of alphas and MSE values,
# justmin=1 prints just the optimum alpha and corresponding MSE.
# tests all integer alphas from 0 to maxalpha.
'''

train1 = np.loadtxt(hw1datapath + 'hw1-data/train-1000-100.csv', delimiter=',', skiprows=1)
train2 = np.loadtxt(hw1datapath + 'hw1-data/train-100-100.csv', delimiter=',', skiprows=1)
train3 = np.loadtxt(hw1datapath + 'hw1-data/train-100-10.csv', delimiter=',', skiprows=1)
test1 = np.loadtxt(hw1datapath + 'hw1-data/test-1000-100.csv', delimiter=',', skiprows=1)
test2 = np.loadtxt(hw1datapath + 'hw1-data/test-100-100.csv', delimiter=',', skiprows=1)
test3 = np.loadtxt(hw1datapath + 'hw1-data/test-100-10.csv', delimiter=',', skiprows=1)


# Take in data from 1000_100_train.csv, create array of first 50, 100, 150 rows, write to csv
newtrain1 = train1[:50]
newtrain2 = train1[:100]       
newtrain3 = train1[:150]

np.savetxt(hw1datapath + 'hw1-data/50(1000) 100 train.csv', newtrain1, delimiter=",")
np.savetxt(hw1datapath + 'hw1-data/100(1000) 100 train.csv', newtrain2, delimiter=",")
np.savetxt(hw1datapath + 'hw1-data/510(1000) 100 train.csv', newtrain3, delimiter=",")
                
# Add first column of 1's to each matrix
def addCol1(matrix):
    Ones = np.ones((matrix.shape[0],1))
    matrix = np.concatenate([Ones[:], matrix[:]], axis=1)
    return matrix

test1 = addCol1(test1)
train1 = addCol1(train1)
test2 = addCol1(test2)
train2 = addCol1(train2)
test3 = addCol1(test3)
train3 = addCol1(train3)

# Take in data from 1000_100_train.csv, create array of first 50, 100, 150 rows
newtrain1 = train1[:50]
newtrain2 = train1[:100]       
newtrain3 = train1[:150]

# Implement L2 regularized linear regression algorithm
# Alpha = learning rate Theta = vector of coefficients

def calcMSE(theta, X, y):
    TotalError = 0
    for i in range(0, len(X)):
        TotalError += (np.dot(np.transpose(theta), np.transpose(X[i])) - y[i])**2
    return TotalError / float(len(X))
               
def L2LR(X, y, alpha): 
    theta = np.dot(np.linalg.inv(np.dot(np.transpose(X), X) + np.dot(alpha, np.eye(X.shape[1]))), np.dot(np.transpose(X), y))
    return theta

def traintest(train, test, alphas): #alphas is list of alphas to check, train is x's only, first col of 1's, test is y vector           
    lenx = train.shape[1]-1
    trainx = train[:,0:lenx]
    trainy = train[:,lenx]
    testx = test[:,0:lenx]
    testy = test[:,lenx]
    #thetaMatrix = np.zeros([len(alphas),len(X[1,:])+1])    #Initialize output: row for each alpha, col for each coefficient and lambda in first col
    mseMatrix = np.zeros([len(alphas), 3])                  #Matrix of alphas col1, trainMSE, testMSE
    for a in alphas:
        thetaOfa = L2LR(trainx, trainy, alpha=a)
       #thetaMatrix[a] = np.append([a], thetaOfa))
        mseMatrix[a] = np.append([a], np.append([calcMSE(thetaOfa, trainx, trainy)], calcMSE(thetaOfa, testx, testy)))
    return mseMatrix

# For each of the 6 dataset, plot both the training set MSE
# and the test set MSE as a function of (x-axis) in one graph.

mod1 = traintest(train1,test1, range(0,151))
mod2 = traintest(train2,test2, range(0,151))
mod3 = traintest(train3,test3, range(0,151))
mod4 = traintest(newtrain1,test1, range(0,151))
mod5 = traintest(newtrain2,test1, range(0,151))
mod6 = traintest(newtrain3,test1, range(0,151))

def optAlpha(arr):
    return arr[np.argmin(arr[:,2]),:]


def plotmse(mod):   #first col on x axis, 2nd and 3rd cols on Y axis #Excludes alpha = 0 for better visualization
    plt.plot(mod[1:,0],mod[1:,1], 'r--', mod[1:,0],mod[1:,2], 'b--')
    plt.xlabel('Lambda')
    plt.ylabel('Mean Squared Error')
    plt.annotate('Train Set', xy=(130, 3.7), xytext=(130, 3.7),color='r')
    plt.annotate('Test Set', xy=(130, 4.3), xytext=(130, 4.3),color='b')
    plt.show()

# plotmse for mod1 through mod6

# Q2

def learningcurve(train, test, alpha): #alphas 1,25,150
    lenx = train.shape[1]-1
    obs = train.shape[0]
    trainx = train[:,0:lenx]
    trainy = train[:,lenx]
    testx = test[:,0:lenx]
    testy = test[:,lenx]
    mseMatrix = []                  #Matrix of trainingsize col1, trainMSE, testMSE
    for i in range(25,525,25):
        randsubset = np.random.choice(obs,size=i,replace=0)         #generate i random numbers between 0 and 1000
        temptrainx = trainx[randsubset,:]    #take random subset of size i from trainx and trainy, same rows from both, calculate theta
        temptrainy = trainy[randsubset]
        thetaOfi = L2LR(temptrainx, temptrainy, alpha)
       #thetaMatrix[a] = np.append([a], thetaOfa))
        mseMatrix.append([i, calcMSE(thetaOfi, testx, testy)])
    return np.asarray(mseMatrix)

    
#repeat 10 learningcurve functions for each alpha, append second columns side by side
alpha1LC = []
alpha25LC = []
alpha150LC = []

for i in range(10):
    alpha1LC.append(learningcurve(train1, test1, 1)[:,1])  # gives np.array 1st col = sample size, 2nd = MSE on test set

for i in range(10):
    alpha25LC.append(learningcurve(train1, test1, 25)[:,1])
    
for i in range(10):
    alpha150LC.append(learningcurve(train1, test1, 150)[:,1])

#Average columns 

alpha1LC = np.mean(alpha1LC,0)
alpha25LC = np.mean(alpha25LC,0)
alpha150LC = np.mean(alpha150LC,0)

xaxis = (list(range(25,525,25)))


def plotLCs():   #first col on x axis, 2nd and 3rd cols on Y axis
    plt.plot(xaxis,alpha1LC,'r--', xaxis, alpha25LC, 'b--', xaxis, alpha150LC, 'g--')
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curve: Predictive Performance as a Function of Training Set Size')
    plt.annotate('Lambda = 1', xy=(400, 10), xytext=(400, 10),color='r')
    plt.annotate('Lambda = 25', xy=(400, 11), xytext=(400, 11),color='b')
    plt.annotate('Lambda = 150', xy=(400, 12), xytext=(400, 12),color='g')
    plt.show()

'''From the plots in question 1, we can tell which value of
 is best for each dataset once we know the test data and its
labels. This is not realistic in real world applications.
In this part, we use cross validation to set the value for .
Implement the CV technique given in the
class slides. For each dataset, compare the values of 
and MSE with the values in question 1). '''

#
#kfolds = cv(k,train)  #alphas is range of lambdas to check, starting at 0
def FindBestLambda(k, train, maxalpha,justmin=1):
    lenx = train.shape[1]-1
    obs = len(train)
    alphas = list(range(0,maxalpha+1))
    randlist = np.random.choice(obs,size=obs,replace=0)     #divide training set into k random pieces
    alphaMatrix = np.zeros([maxalpha+1,2]) #row for each alpha, col for alpha, and average MSE
    for a in alphas:
        mseMatrix = np.zeros([k, 1])    #row for each fold, col for testMSE
        for i in range(1,k+1):        #returns key, reference values
            temptestx = train[randlist[((i-1)*(obs/k)):(i*(obs/k))],0:lenx] #how to combine all arrays into one? getkfolds
            temptesty = train[randlist[((i-1)*(obs/k)):(i*(obs/k))],lenx]
            if i == 1:
                temptrainx = train[randlist[((i)*(obs/k)):(obs)],0:lenx]
                temptrainy = train[randlist[((i)*(obs/k)):(obs)],lenx]
            else:
                temptrainx2 = train[randlist[((i)*(obs/k)):(obs)],0:lenx]
                temptrainy2 = train[randlist[((i)*(obs/k)):(obs)],lenx]
                temptrainy1 = train[randlist[((i-2)*(obs/k)):(((i-1)*(obs/k)))],lenx]
                temptrainx1 = train[randlist[((i-2)*(obs/k)):(((i-1)*(obs/k)))],0:lenx]
                temptrainx = np.concatenate((temptrainx1,temptrainx2))
                temptrainy = np.concatenate((temptrainy1,temptrainy2))
            thetak = L2LR(temptrainx, temptrainy, a)
            mseMatrix[i-1] = calcMSE(thetak, temptestx, temptesty) #append testMSE in mseMatrix,
        alphaMatrix[a] = [a,np.average(mseMatrix)]                                  #record MSE and alpha in alphamatrix
    if justmin == 1:
        minMSE = alphaMatrix[np.argmin(alphaMatrix[:, 1])]
        print('Minimum MSE is obtained when lambda = ' + str(minMSE[0])+'.' ' MSE = ' + str(minMSE[1]))
    else:
        print(alphaMatrix)

           



print('''
QUICKSTART 
Training - Testing Pairs
1. train1     test1
2. train2     test2
3. train3     test3
4. newtrain1  test1
5. newtrain2  test1
6. newtrain3  test1

Key Functions
Q1
addCol1(matrix)  adds column of ones at beginning of data matrix, for intercept values
calcMSE(theta, X, y)
L2LR(X, y, alpha)
traintest(train, test, alphas): #Creates matrix of alphas and testing MSEs for range of alphas, eg alphas = range(0,151)
plotmse(mod) : takes in matrix from traintest() function and plots alphas(x) vs. MSE 

Q2
plotLCs() plots learning curves for alphas 1,25,150, LCs are averaged after 10 trials each.

Q3
FindBestLambda(k, trainingset, maxalpha,justmin=1)
# Set justmin=0 to get table of alphas and MSE values,
# justmin=1 prints just the optimum alpha and corresponding MSE.
# tests all integer alphas from 0 to maxalpha.
END QUICKSTART
''')
