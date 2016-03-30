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
n_sample = 1000
thres = 0.9
X = np.random.rand(n_sample)
X[X<thres] = 0.0
X[X>thres] = 1.0

Y = []
Y.append(X[0])
for i in range(1,n_sample):
    Y.append(r*Y[-1] + X[i] - teta*X[i-1])
Y = np.array(Y)

plt.figure("expl: ARMA(1,1)")
plt.plot(Y)

#%% Partition for X

def G(orde,X):
    """
    arrondi à l'odre "orde" toute les valeurs de X
    """
    div = 10^orde
    return np.round(div*X)/div


#G_y = G(2,Y)
#
#plt.figure()
#plt.plot(G_y)
#plt.plot(Y)

#%% Elementary predictor

def h(k,orde,X,Y_n_1):
    """
    Elementary predictor of the paper 
    """
    n_sample = len(X)
    if(2*k<n_sample):
        Y = G(orde,Y_n_1)
        last_seq_Y = Y[n_sample-k:-1]
        last_seq_X = X[n_sample-k:]
        count = 0
        val = 0
        for i in range(n_sample-2*k):
            if((X[i:i+k].all()==last_seq_X.all())and(Y[i:i+k-1].all()==last_seq_Y.all())):
                count += 1.0
                val += Y[i+k]
        if(count == 0):
            out = 0.0
        else:
            out = val/count
    else:
        out = 0.0
    
    return out

#print "Err: ", (h(1,1,X,Y)-Y[-1])**2

#%% Matrices

k_range = range(1,4)
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
    for k in k_range:
        for l in l_range:
            pred_mat[k-1,l-1] = h(k,l,X_tronc,Y_tronc)
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
plt.plot(pred)
plt.plot(Y[10:])














