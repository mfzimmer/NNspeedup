#author: Michael Zimmer


import numpy as np


# --------------------------------------------------------------
def randomUnitVector( dim1 ):
    # Since I'm normalizing the vector, require mu=0
    mu=0.0
    sigma=1.0
    s = np.random.normal(mu, sigma, dim1)
#    print 's = ', s
    radius = np.sqrt( np.sum( [x*x for x in s]) )
#    print 'radius = ', radius
    return np.array([x / radius for x in s])


# --------------------------------------------------------------
# Create a set of random weights w, to be used between two layers in a NN.
# Creates dim2 arrays of unit vectors, each one being array in dim1 dimensions
def randomWeights(dim1, dim2):
    w = []
    for i in range(dim2):
        w.append( randomUnitVector(dim1) )
    return np.array(w)


# --------------------------------------------------------------        
# b_01 = use 01 version
def softmax(y,b_01):
    if b_01:
        return 1/(1.0 + np.exp(-y))         # for 0,1 version
    else:
        return 2.0/(1.0 + np.exp(-y)) - 1.0  # for -1,1 version
        

# --------------------------------------------------------------        
def softmax_01(y):
    return 1.0/(1.0 + np.exp(-y))  # for 0,1 version

# --------------------------------------------------------------        
def f_der(y,b_01):
    if b_01:
        return softmax_01(y)*(1 - softmax_01(y))  # for -1,1 version
    else:
        return 2*softmax_01(y)*(1 - softmax_01(y))  # for -1,1 version
#    return softmax_old(y)*(1 - softmax_old(y))   # for 0,1 version


# --------------------------------------------------------------        
def getError( a,b ):
    # check if lengths of list are the same
    # if len(a) != len(b):
    #       print 'ERROR in function getError
    sum = 0.0
    for i in range(len(a)):
        sum += (a[i] - b[i]) * (a[i] - b[i])
    return sum/2.0


#---------------------------------------------------------------
def getCostFcn(data, target_data, w1, w2, d2, d3, c1, c2, s1, s2):
    n2 = np.zeros(d2)
    n3 = np.zeros(d3)
    err_sum = 0.0
    for k in range(len(data)):
        for i in range(d2):
            h = s1[i] * ( np.dot(w1[i], data[k]) + c1[i] )
            n2[i] = softmax(h)

        for j in range(d3):
            h = s2[j] * ( np.dot(w2[j], n2) + c2[j] )
            n3[j] = softmax(h)

        err_sum += getError( n3, target_data[k] )
    return(err_sum)


#---------------------------------------------------------------
# Return input and target data for auto-encoding example.  They're independent.      
def getAutoEncData(d1, b_01):
    # Identity array d1-by-d1
    dat  = np.eye(d1, dtype=float)

    # If not using 0/1 encoding, convert to -1/1 encoding
    if(not b_01):
        dat  = 2*dat - np.ones(np.shape(dat))

    return(dat,dat.copy())

#---------------------------------------------------------------
# Returns the mean (cm) and standard dev (sd) for a data set 'dat'
def getStats(dat):
    # N = number of data points
    N = len(dat)
    # nf = number of features in each data point; require N >= 1
    nf = len(dat[0])  
    cm     = np.zeros(nf)
    sum_sq = np.zeros(nf)
    for d in dat:
        cm += d
        sum_sq += np.array([x*x for x in d])
    cm /=  N
    sd = np.sqrt( sum_sq / N - cm * cm )  # sd for population, not sample
    return(cm,sd)

#---------------------------------------------------------------
# Shift and scale the data based on its CM and SD.  Result should then have CM=0 and SD=1
def scaleData(dat,cm,sd):
    y = np.max( np.abs(dat), axis=0)  # max feature over all samples

    # If every element is the same non-zero value, divide by its magnitude instead
    for i in range(len(sd)):
        if sd[i] == 0:
            sd[i] = y[i]
    
    # TO DO: raise an error in this case
    for i in range(len(sd)):
        if sd[i] == 0:
            pass

    return( (dat - cm) / sd)  # this will be an independent, altered copy of the data
    
    

