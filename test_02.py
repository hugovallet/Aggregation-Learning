# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 12:42:37 2016

@author: ivaylo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 18:54:45 2016

@author: ivaylo
"""

import numpy as np
import matplotlib.pyplot as plt

#%% ARMA(1,1)

r = 0.6
teta = 0.2
n_sample = 2000
thres = 0.95
X = np.random.rand(n_sample)
X[X<thres] = 0.0
X[X>thres] = 1.0

Y = []
Y.append(X[0])
for i in range(1,n_sample):
    Y.append(r*Y[-1] + X[i] - teta*X[i-1])
Y = np.array(Y)

plt.figure()
plt.title("expl: ARMA(1,1)")
plt.plot(Y)

#%% ARMA(2,1)

r1 = 0.5
r2 = 0.3
teta = 0.2
n_sample = 1000
thres = 0.95
X = np.random.rand(n_sample)
X[X<thres] = 0.0
X[X>thres] = 5.0

Y = []
Y.append(X[0])
Y.append(r1*Y[-1] - teta*X[0])
for i in range(2,n_sample):
    Y.append(r1*Y[-1] + r2*Y[-2] + X[i] - teta*X[i-1])
Y = np.array(Y)

plt.figure()
plt.title("expl: ARMA(2,1)")
plt.plot(Y)

#%% Partition for X

def G(orde,X):
    """
    arrondi à l'odre "orde" toute les valeurs de X
    """
    div = 10**orde
    return np.round(div*X)/div


#G_y = G(2,Y)
#
#plt.figure()
#plt.plot(G_y)
#plt.plot(Y)

#%% Elementary predictor

def h(k,orde,X,Y_n_1):
    """#%% Visualization

    Elementary predictor of the paper 
    """
    n_sample = len(X)
    if(2*k<n_sample):
        Y = G(orde,Y_n_1)
        last_seq_Y = Y[-k+1:]
        last_seq_X = X[-k:]
        count = 0
        val = 0
        for i in range(n_sample-2*k):
            if((list(X_tronc[i:i+k])==list(last_seq_X))and(list(Y_tronc[i:i+k-1])==list(last_seq_Y))):
                count += 1.0
                val += Y[i+k-1]
        if(count == 0):
            out = 0.0
        else:
            out = val/count
    else:
        out = 0.0
    
    return out

#print "Err: ", (h(1,1,X,Y)-Y[-1])**2

#%% Matrices

k_range = range(2,3)
l_range = range(1,2)

# Predictions individuelles
pred_mat = np.zeros((len(k_range),len(l_range)))
# Erreur L_t pour chaque elementary pred
err_mat = np.zeros((len(k_range),len(l_range)))
# Poids pour prédiction finale
weigths_mat = np.ones((len(k_range),len(l_range)))
# Normaliser les weigths
weigths_mat = weigths_mat/np.sum(weigths_mat)


time = 0
pred = []
err = []
for n in range(10,n_sample):
    X_tronc = X[:n]
    Y_tronc = Y[:n-1]
    target = Y[n]
    for k in range(len(k_range)):
        for l in range(len(l_range)):
            #print  h(k_range[k],l_range[l],X_tronc,Y_tronc)           
            pred_mat[k-1,l-1] = h(k_range[k],l_range[l],X_tronc,Y_tronc)
    time += 1.0
    # Weigths update
    weigths_mat = np.exp((-(time-1.0)/np.sqrt(time))*err_mat)
    weigths_mat = weigths_mat/np.sum(weigths_mat)
    # Prediction and Error
    combined_pred = np.sum(weigths_mat*pred_mat)
    pred.append(combined_pred)
    err.append((combined_pred-target)**2)
    # Update of err_mat for weigths
    err_mat = ((time-1.0)/time)*err_mat + (pred_mat - target)**2/time 
    if(n%100==0):
        print "step: ",n

print "End"

#%% Visualization

plt.figure()
plt.title("Algorithm for ARMA(1,1)")
plt.plot(pred,label="Prediction")
plt.plot(Y[9:],label="True Value")
plt.legend(loc=0)

#%% Test function

n = 1382
k = 2
orde = 1

X_tronc = X[:n]
Y_tronc = Y[:n-1]
Y_tronc2 = G(orde,Y_tronc)
target = Y[n]

n_sample = len(X_tronc)
if(2*k<n_sample):
    Y_tronc = G(orde,Y_tronc)
    last_seq_Y = Y_tronc[-k+1:]
    last_seq_X = X_tronc[-k:]
    count = 0
    val = 0
    for i in range(n_sample-2*k):
        if((list(X_tronc[i:i+k])==list(last_seq_X))and(list(Y_tronc[i:i+k-1])==list(last_seq_Y))):
            count += 1.0
            val += Y_tronc[i+k-1]
            print Y_tronc[i+k-1]
            #print val/count
    if(count == 0):
        out = 0.0
    else:
        out = val/count
else:
    out = 0.0





