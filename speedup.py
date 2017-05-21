# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:46:13 2017


author: Michael Zimmer
"""

from utils_net import randomWeights, softmax, getError, f_der, getAutoEncData, getStats, scaleData

import numpy as np
import matplotlib.pyplot as plt
import time

time_start = time.clock()


# random seed ---------------------------------------------
np.random.seed(423418)

# Encoding ----------------------------------------------
# True: encode data with 0,1
# False encode data wit -1, 1b (works better!)
bUse_01 = False

# Auto-encoding (d1-d2-d3) ---------------------------------
# Use: (8,3), (16,4), (32,5), (64,6), (128,7)
d1 = 16
d2 = 4
d3 = d1

# Get input (in_data) and target data (t)
in_data,t = getAutoEncData(d1, bUse_01)

# Obtain the mean (CM) and stan dev (SD) along each of the d1 axes.  They don't have to be the same along each axis in general.
n1_CM,n1_SD = getStats(in_data)

# The data now has zero mean and unit SD.  Note that 'data' is independent from 'in_data'
data = scaleData( in_data, n1_CM, n1_SD )


# PARAMETERS ==============================================================

# Use these d1-dependent eta values in the section below
# d_1  &  eta_Z   & eta_W
# 8    &  0.18    &  0.45
# 16   &  0.11    &  0.22
# 32   &  0.04    &  0.12
# 64   &  0.016   &  0.07
# 128  &  0.01    &  0.06   

# If true, use z-param.  Otherwise it uses w-param
b_useZparam = False

if b_useZparam:
    # New Z-param -------------------------
    s_mult = 1      # Normally it's 1.0, when doing my new Zparam
    eta    = 0.11      # learning rate  (0.125 for d1=16)
    model_name = "Z"
else:    
    # W-param -------------------------
    s_mult = 0      # This sets the s variable to 0.0, since we use s * s_mult
    eta    = 0.22      # learning rate  (0.26 for d1=16)
    model_name = "W"


num_loops = 5   # number of runs
num_iter = 800  # number of epochs
chk_freq = 5    # number of epochs before recording error rate
if int(num_iter / chk_freq) != (num_iter / chk_freq):
    print 'ERROR with choice of num_iter and chk_freq'
    
n_points = int(num_iter / chk_freq)     # number of error measurements
print 'n_points = ', n_points

# Sets the size of h for initialization (multiplies s for z-param, multiplies u for w-param)
s_scale = 0.1


# VARIABLES ==============================================================

# Arrays for plotting ----------------------------
n_plt = np.zeros( (num_loops,n_points) )    # epoch number
e_plt = np.zeros( (num_loops,n_points) )    # error

# Initializations --------------------------------
#err_sum_old = 0.0
err_sum_avg = 0
err_sum_cnt = 0
der_s1 = np.zeros(d2)
der_s2 = np.zeros(d3)
der_c1 = np.zeros(d2)
der_c2 = np.zeros(d3)
der_u1 = np.zeros( (d2,d1) )	
der_u2 = np.zeros( (d3,d2) )
n2 = np.zeros(d2)
n3 = np.zeros(d3)
c1 = np.zeros(d2)
c2 = np.zeros(d3)
h1 = np.zeros(d2)
h2 = np.zeros(d3)
Z = np.zeros(d3)


# FILE I/O ============================================================
filename = "outMain_{}_eta{}_loops{}_iter{}_s{}_d{}.txt".format(model_name,eta,num_loops,num_iter,s_scale,d1)
target = open(filename, 'w')
# write output---------------------------
target.write("# comment: eta={}\n".format(eta) )
target.write("d1 {}\n".format(d1) )
target.write("num_loops {}\n".format(num_loops) )
target.write("n_points {}\n".format(n_points) )



# LOOPS ==================================================================

# In this program, go over the data in one batch.
# Do "num_iter" iterations over this one batch, which means "num_iter" epochs for the NN.
# Also, do this "num_loops" times, for averaging purposes.

for nn in range(num_loops):
    print ' -------------------- loop # ', nn, ' ----------------------------'

    u1 = randomWeights( d1, d2 )
    u2 = randomWeights( d2, d3 )
    n2 = 0*n2
    n3 = 0*n3
    c1 = 0*c1
    c2 = 0*c2
    s1 = np.ones(d2)
    s2 = np.ones(d3)
    h1 = 0*h1
    h2 = 0*h2    
    Z = 0*Z    
    u1_mag = np.zeros(d2)
    u2_mag = np.zeros(d3)
    idx = 0
    # This is for scaling s in z-param case, or equivalent in w-param case
    if b_useZparam:
        s1 *= s_scale
        s2 *= s_scale
    if not b_useZparam:
        u1 *= s_scale
        u2 *= s_scale

    for n in range(num_iter):        
        der_s1 = 0*der_s1  # 1st layer
        der_c1 = 0*der_c1
        der_u1 = 0*der_u1
        der_s2 = 0*der_s2  # 2nd layer
        der_c2 = 0*der_c2
        der_u2 = 0*der_u2
        
        # Go over ALL the data for each update-----------------------------------    
        for kk in range(len(data)):
    
            # NN pass-thru --------------------------------------------
            for j in range(d2):
                h1[j] = s1[j] * ( np.dot(u1[j], data[kk]) + c1[j] )
                n2[j] = softmax(h1[j],bUse_01)
            for k in range(d3):
                # Use n2 CM here...
                h2[k] = s2[k] * ( np.dot(u2[k], n2) + c2[k] )
                n3[k] = softmax(h2[k],bUse_01)
    
            # "back prop" -------------------------------------------- layer 2/3
            for k in range(d3):
                Z[k] = f_der(h2[k],bUse_01) * (n3[k] - t[kk][k])
                der_s2[k] += (c2[k] + np.dot(u2[k], n2)) * Z[k]
                der_c2[k] += s2[k] * Z[k]
                for j in range(d2):
                    der_u2[k][j] += s2[k] * n2[j] * Z[k]
    
            # "back prop" -------------------------------------------- layer 1/2
            for j in range(d2):
                sum = 0
                for k in range(d3):
                    sum += s2[k] * u2[k][j] * Z[k]
                Y = f_der(h1[j],bUse_01) * sum
                der_s1[j] += (c1[j] + np.dot(u1[j], data[kk])) * Y
                der_c1[j] += s1[j] * Y
                for i in range(d1):
                    der_u1[j][i] += s1[j] * data[kk][i] * Y
    
        # This spearates z-param from w-param   
        # If s_mult==0, s can never be updated.  It stays at its initial value.
        der_s1 *= s_mult
        der_s2 *= s_mult
        
    
        # Update params -----------------------------------------------    
        for k in range(d3):
            s2[k] -= eta * der_s2[k]
            c2[k] -= eta * der_c2[k]
            for j in range(d2):
                u2[k][j] -= eta * der_u2[k][j]
        
        for j in range(d2):
            s1[j] -= eta * der_s1[j]
            c1[j] -= eta * der_c1[j]
            for i in range(d1):
                u1[j][i] -= eta * der_u1[j][i]
        
    
        # Compute new cost fcn-----------------------------------------    
        err_sum = 0
        for kk in range(len(data)):
            for j in range(d2):
                h1[j] = s1[j] * ( np.dot(u1[j], data[kk]) + c1[j] )
                n2[j] = softmax(h1[j],bUse_01)
            for k in range(d3):
                h2[k] = s2[k] * ( np.dot(u2[k], n2) + c2[k] )
                n3[k] = softmax(h2[k],bUse_01)
            err_sum += getError( n3, t[kk] )
        err_sum_old = err_sum
    
        # -----------------------------------------------    
        if n % chk_freq == chk_freq -1:
            print 'n,err_sum_old = ', n, err_sum_old
            n_plt[nn][idx] = n
            e_plt[nn][idx] = err_sum_old
            # write output----------------
            st = "{0} {1}\n".format(n, err_sum_old)
            target.write(st)            
            # ----------
            idx += 1


    # Note that 'err_sum_avg' uses the last  
    err_sum_avg += err_sum_old
    err_sum_cnt += 1


    # timer
    print 'time elapsed (min) to here = ', (time.clock() - time_start)/60
    print 'time elapsed (sec) to here = ', (time.clock() - time_start)


# ====================================================
print 'avg error (last entry) = ', err_sum_avg / (1.0 * err_sum_cnt)
print 'err_sum_cnt = ', err_sum_cnt


# close output file -------------------------------
target.close()




# Do plotting =========================================================
for nn in range(num_loops):
    # skip first 2
    plt.plot( n_plt[nn][2:], e_plt[nn][2:] )
plt.show()



print 'total time elapsed (min) = ', (time.clock() - time_start)/60
print 'total time elapsed (sec) = ', (time.clock() - time_start)

