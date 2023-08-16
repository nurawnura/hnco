import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf
from sklearn.metrics import silhouette_samples


def no_complex_num(complex_num):
    complex_num= np.array(complex_num)
    shape = complex_num.shape
    if len(list(shape)) > 1:
        shape_new = np.prod(list(shape))
        complex_num= complex_num.reshape(1, shape_new)

    new_num = []
    for i in range(len(complex_num)):
        new_num.append(complex_num[i].real)
    new_num = np.array(new_num).reshape(shape)
    return new_num


def marsenko_pastur_PDF(var, q, pts):
    eMin, eMax = var*(1-(1./q)**.5)**2, var*(1+(1./q)**.5)**2
    eVal = np.linspace(eMin, eMax, pts) #eVal='lambda'
    pdf = q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5 
    pdf = pd.Series(pdf, index=eVal)
    return pdf


def getPCA(matrix):
    eVal, eVec = np.linalg.eig(matrix) #complex Hermitian (conjugate symmetric) or a real symmetric matrix.
    indices = eVal.argsort()[::-1]
    eVal,eVec = eVal[indices],eVec[:,indices]
    eVal = np.diagflat(eVal)
    return eVal, eVec
    
    
def fitKDE(obs, bWidth=0.15, kernel='gaussian', x=None):
    if len(obs.shape) == 1: 
        obs = obs.reshape(-1,1)
    kde = KernelDensity(kernel = kernel, bandwidth = bWidth).fit(obs)
    if x is None: 
        x = np.unique(obs).reshape(-1,1)
    if len(x.shape) == 1: 
        x = x.reshape(-1,1)
    logProb = kde.score_samples(x)
    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf


def cov2corr(cov_matrix):
    Dinv = np.diag(1 / np.sqrt(np.diag(cov_matrix))) 
    corr_matrix = Dinv @ cov_matrix @ Dinv
    corr_matrix[corr_matrix<-1], corr_matrix[corr_matrix>1] = -1,1 #for numerical errors
    return corr_matrix
    
    
def corr2cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov     
    
    
def error_PDFs(var, eVal, q, bWidth, pts=1000):
    var = var[0]
    pdf0 = marsenko_pastur_PDF(var, q, pts)
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values) 
    sse = np.sum((pdf1-pdf0)**2)
    return sse 

    
def findMaxEval(eVal, q, bWidth):
    out = minimize(lambda *x: error_PDFs(*x), x0=np.array(0.5), 
                   args=(eVal, q, bWidth), bounds=((1E-5, 1-1E-5),))
    if out['success']: 
        var = out['x'][0]
    else: 
        var=1
    eMax = var*(1+(1./q)**.5)**2
    return eMax, var
    
    
def denoisedCorr(eVal, eVec, nFacts):
    eVal_ = np.diag(eVal).copy()
    eVal_[nFacts:] = eVal_[nFacts:].sum()/float(eVal_.shape[0] - nFacts)
    eVal_ = np.diag(eVal_) 
    corr1 = np.dot(eVec, eVal_).dot(eVec.T) #Eigendecomposition of a symmetric matrix: S = QΛQT
    corr1 = cov2corr(corr1)
    return corr1


def deNoiseCov(cov0, q, bWidth):
    corr0 = cov2corr(cov0)
    eVal0, eVec0 = getPCA(corr0)
    eVal0, eVec0 = no_complex_num(eVal0), no_complex_num(eVec0)
    eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth)
    nFacts0 = eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0)
    corr1 = denoisedCorr(eVal0, eVec0, nFacts0) #denoising by constant residual eigenvalue method
    cov1 = corr2cov(corr1, np.diag(cov0)**.5)
    return cov1


def optPort(cov, mu = None):
    inv = np.linalg.inv(cov) 
    ones = np.ones(shape = (inv.shape[0], 1))
    if mu is None: 
        mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w) 
    
    return w 


def minVarPort(cov):
    return optPort(cov, mu = None)


def CovMatCondNum(matrix, symmetric=True):
    if symmetric == True:
        return abs(max(np.linalg.eigh(matrix)[0])/min(np.linalg.eigh(matrix)[0]))
    
    
def clstrdMatrix(corr_df, clstrs):
    array = []
    for i in clstrs:
        array += list(clstrs[i])
    
    return corr_df.loc[array, array], array


def array2df(array, columns, index=None):
    df = pd.DataFrame()
    i = 0
    if index is None:
        index = columns
    for col in columns:
        df[col] = array.T[i]
        i += 1
    df.index = index
    return df


def clstrs_to_join(reduced_corr, threshold=0.25):
    np_redcorr = np.array(reduced_corr)
    np_redcorr[np.diag_indices_from(np_redcorr)] = 0
    npr_df = pd.DataFrame(np_redcorr, columns=reduced_corr.columns, index=reduced_corr.index)
    upper_npr = []
    for i in range(len(npr_df)):
        upper_npr.append(list(npr_df[i][:i]) + [0]*(len(npr_df)-i))
    upper_npr = pd.DataFrame(upper_npr, index=npr_df.index, columns=npr_df.columns)
    upper_npr = upper_npr.T
    join = []
    k = 0
    for i in upper_npr.index:
        row = upper_npr.loc[i]
        ind = list(row[row>threshold].T.index)
        if len(ind) > 0:
            join.append([i] + ind)
            upper_npr = upper_npr.drop(ind, axis=1)
        k += 1
    return join


def join_clstrs(clstrs, join, del_noise=True):
    new_clstrs = clstrs
    if del_noise:
        noise = new_clstrs.pop(-1)
    if len(join) > 0:
        for i in join:
            new_clstrs[i[0]] += new_clstrs[i[1]]
        for i in join:
            pop_i = new_clstrs.pop(i[1])
    return new_clstrs   


def clstrs_returns(returns, clstrs, weights):
    clstrs_returns = []
    for i in clstrs:
        mean_clstr_returns = np.array(returns[clstrs[i]]).mean(axis=0)
        clstrs_returns.append(np.array(weights[i]).dot(mean_clstr_returns))
    return clstrs_returns 