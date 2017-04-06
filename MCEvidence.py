#!usr/bin/env python
"""
Version : 0.1.1
Date : 1st March 2017

Authors : Yabebal Fantaye
Email : yabi@aims.ac.za
Affiliation : African Institute for Mathematical Sciences - South Africa
              Stellenbosch University - South Africa

License : MIT

Status : Under Development

Description :
Python2.7 implementation of the evidence estimation from MCMC chains 
as preesented in A. Heavens et. al. 2017
(paper can be found here : https://arxiv.org/abs/ ).
"""

from __future__ import absolute_import
from __future__ import print_function
import subprocess
import importlib
import itertools
from functools import reduce
import io

import tempfile 
import os
import sys
import math
import numpy as np
import pandas as pd
import sklearn as skl
import statistics
from sklearn.neighbors import NearestNeighbors, DistanceMetric
import scipy.special as sp
from numpy.linalg import inv
from numpy.linalg import det


__author__ = "Yabebal Fantaye"
__email__ = "yabi@aims.ac.za"
__license__ = "MIT"
__version__ = "0.1.1"
__status__ = "Development"

try:
    from getdist import MCSamples, chains
    from getdist import plots, IniFile
    import getdist as gd
    
    #====================================
    #      Getdist wrapper
    #====================================    
    class samples2gdist(object):
        #Ref:
        #http://getdist.readthedocs.io/en/latest/plot_gallery.html

        def __init__(self,str_or_dict,trueval=None,
                    names=None,labels=None,px='x',**kwargs):
            #Get the getdist MCSamples objects for the samples, specifying same parameter
            #names and labels; if not specified weights are assumed to all be unity

            if isinstance(str_or_dict,str):
                
                fileroot=str_or_dict
                print('samples2gdist: string passed. Loading chain from '+fileroot)                
                self.load_from_file(fileroot,**kwargs)
                
            elif isinstance(str_or_dict,dict):
                
                d=str_or_dict
                chains=d['chains']
                loglikes=d['loglikes']
                weight=d['weight'] if 'weight' in d.keys() else np.ones(len(loglikes))

                if names is None:
                    names = ["%s%s"%('p',i) for i in range(ndim)]
                if labels is None:
                    labels =  ["%s_%s"%(px,i) for i in range(ndim)]
                    
                self.names=names
                self.labels=labels
                self.trueval=trueval
                self.samples = gd.MCSamples(samples=chain,
                                            loglikes=lnprob,
                                            weight=weight,
                                            names = names,
                                            labels = labels)
            else:               
               print('first argument to samples2getdist should be a string or dict.')
               print('Passed first argument type is: ',type(str_or_dict))
               

        def triangle(self,**kwargs):
            #Triangle plot
            g = gd.plots.getSubplotPlotter()
            g.triangle_plot(self.samples, filled=True,**kwargs)

        def plot_1d(self,l,**kwargs):
            #1D marginalized plot
            g = gd.plots.getSinglePlotter(width_inch=4)        
            g.plot_1d(self.samples, l,**kwargs)

        def plot_2d(self,l,**kwargs):
            #Customized 2D filled comparison plot
            g = gd.plots.getSinglePlotter(width_inch=6, ratio=3 / 5.)       
            g.plot_1d(self.samples, l,**kwargs)      

        def plot_3d(self,llist):
            #2D scatter (3D) plot
            g = gd.plots.getSinglePlotter(width_inch=5)
            g.plot_3d(self.samples, llist)   

        def save_to_file(self,path=None,dname='chain',froot='test'):
            #Save to file
            if path is None:
                path=tempfile.gettempdir()
            tempdir = os.path.join(path,dname)
            if not os.path.exists(tempdir): os.makedirs(tempdir)
            rootname = os.path.join(tempdir, froot)
            self.samples.saveAsText(rootname)     

        def load_from_file(self,rootname,**kwargs):
            #Load from file
            #self.samples=[]
            #for f in rootname:
            self.samples=gd.loadMCSamples(rootname,**kwargs)
                
        def thin(self,nminwin=1,nthin=None):
            if nthin is None:
                ncorr=max(1,int(self.samples.getCorrelationLength(nminwin)))
            else:
                ncorr=nthin
            print('ncorr=',ncorr)
            try:
                self.samples.thin(ncorr)
            except:
                print('Thinning not possible. Weight must be interger to apply thinning.')

        def arrays(self):            
            s=self.samples.samples
            lnp=-self.samples.loglikes
            w=self.samples.weights
            return s, lnp, w
            
        def info(self): 
            #these are just to show getdist functionalities
            print(self.samples.PCA(['x1','x2']))
            print(self.samples.getTable().tableTex())
            print(self.samples.getInlineLatex('x1',limit=1))

except:    
    print('getdist is not installed. You can not use the wrapper: samples2gdist')                      
    raise


#============================================================
#======  Here starts the main Evidence calculation code =====
#============================================================

class MCEvidence(object):
    def __init__(self,method,ischain=True,
                     thin=True,nthin=None,
                     ndim=None,burnin=0.2,
                     nsample=None,
                      nbatch=1,
                      brange=None,
                      bscale='',
                      kmax= 5,        
                      args={},                                            
                      gdkwarg={},
                      verbose=1):
        """Evidence estimation from MCMC chains
        :param method: chain name (str) or array (np.ndarray) or python class
                If string or numpy array, it is interpreted as MCMC chain. 
                Otherwise, it is interpreted as a python class with at least 
                a single method sampler and will be used to generate chain.

        :param ischain (bool): True indicates the passed method is to be interpreted as a chain.
                This is important as a string name can be passed for to 
                refer to a class or chain name 

        :param nbatch (int): the number of batchs to divide the chain (default=1) 
               The evidence can be estimated by dividing the whole chain 
               in n batches. In the case nbatch>1, the batch range (brange) 
               and batch scaling (bscale) should also be set

        :param brange (int or list): the minimum and maximum size of batches in linear or log10 scale
               e.g. [3,4] with bscale='logscale' means minimum and maximum batch size 
               of 10^3 and 10^4. The range is divided nbatch times.

        :param bscale (str): the scaling in batch size. Allowed values are 'log','linear','constant'/

        :param kmax (int): kth-nearest-neighbours, with k between 1 and kmax-1

        :param args (dict): argument to be passed to method. Only valid if method is a class.
        
        :param gdkwarg (dict): arguments to be passed to getdist.

        :param verbose: chattiness of the run
        
        """
        #
        self.verbose=verbose
        self.info={}
        #
        self.nbatch=nbatch
        self.brange=brange #todo: check for [N] 
        self.bscale=bscale if not isinstance(self.brange,int) else 'constant'
        
        # The arrays of powers and nchain record the number of samples 
        # that will be analysed at each iteration. 
        #idtrial is just an index
        self.idbatch=np.arange(self.nbatch,dtype=int)
        self.powers  = np.zeros(self.nbatch)
        self.bsize  = np.zeros(self.nbatch,dtype=int)
        self.nchain  = np.zeros(self.nbatch,dtype=int)               
        #
        self.kmax=max(2,kmax)
        #
        self.ischain=ischain
        #
        self.fname=None
        #
        if ischain:
            
            if isinstance(method,str):
                self.fname=method      
                print('Using chains: ',method)
            else:
                print('dictionary of samples and loglike array passed')
                
        else: #python class which includes a method called sampler
            
            if nsample is None:
                self.nsample=100000
            else:
                self.nsample=nsample
            
            #given a class name, get an instance
            if isinstance(method,str):
                print('my method',method)
                XClass = getattr(sys.modules[__name__], method)
            else:
                XClass=method
            
            if hasattr(XClass, '__class__'):
                print('eknn: method is an instance of a class')
                self.method=XClass
            else:
                print('eknn: method is class variable .. instantiating class')
                self.method=XClass(*args)                
                #if passed class has some info, display it
                try:
                    print()
                    msg=self.method.info()                        
                    print()
                except:
                    pass                        
                # Now Generate samples.
                # Output should be dict - {'chains':,'logprob':,'weight':} 
                method=self.method.Sampler(nsamples=self.nsamples)                                 
                
        #======== By this line we expect only chains either in file or dict ====
        self.gd = samples2gdist(method,**gdkwarg)
        self.nparamMC=self.gd.samples.paramNames.numNonDerived()
        if ndim is None: ndim=self.nparamMC
        self.ndim=ndim
        
        #get information about input samples
        sample_shape=self.gd.samples.samples.shape
        npar=sample_shape[-1]
        
        #
        self.info['Nsamples_read']=sample_shape[0]
        self.info['Nparams_read']=npar
        self.info['NparamsMC']=self.nparamMC
        self.info['NparamsCosmo']=self.ndim        
        self.info['MaxAutoCorrLen']=np.array([self.gd.samples.getCorrelationLength(j) for j in range(self.ndim)]).max()

        #print('***** ndim,nparamMC,MaxAutoCorrLen :',self.ndim,self.nparamMC,self.info['MaxAutoCorrLen'])
        
        if thin:
            _=self.gd.thin(nthin=nthin)

        if burnin>0:
            self.gd.samples.removeBurn(remove=0.3)
        #
        self.nsample=self.gd.samples.samples.shape[0]
        self.info['Nsamples']=self.nsample #after thinning
        
        #print('init minmax logl',method['lnprob'].min(),method['lnprob'].max())            
        print('chain array dimensions: %s x %s ='%(self.nsample,self.ndim))
            
        #
        self.set_batch()


    def summary(self):
        print()
        print('ndim={}'.format(self.ndim))
        print('nsample={}'.format(self.nsample))
        print('kmax={}'.format(self.kmax))
        print('brange={}'.format(self.brange))
        print('bsize'.format(self.bsize))
        print('powers={}'.format(self.powers))
        print('nchain={}'.format(self.nchain))
        print()
        
    def get_batch_range(self):
        if self.brange is None:
            powmin,powmax=None,None
        else:
            powmin=np.array(self.brange).min()
            powmax=np.array(self.brange).max()
            if powmin==powmax and self.nbatch>1:
                print('nbatch>1 but batch range is set to zero.')
                raise
        return powmin,powmax
    
    def set_batch(self,bscale=None):
        if bscale is None:
            bscale=self.bscale
        else:
            self.bscale=bscale
            
        #    
        if self.brange is None: 
            self.bsize=self.brange #check
            powmin,powmax=None,None
            self.nchain[0]=self.nsample
            self.powers[0]=np.log10(self.nsample)
        else:
            if bscale=='logpower':
                powmin,powmax=self.get_batch_range()
                self.powers=np.linspace(powmin,powmax,self.nbatch)
                self.bsize = np.array([int(pow(10.0,x)) for x in self.powers])
                self.nchain=self.bsize

            elif bscale=='linear':   
                powmin,powmax=self.get_batch_range()
                self.bsize=np.linspace(powmin,powmax,self.nbatch,dtype=np.int)
                self.powers=np.array([int(log10(x)) for x in self.nchain])
                self.nchain=self.bsize

            else: #constant
                self.bsize=self.brange #check
                self.powers=self.idbatch
                self.nchain=np.array([x for x in self.bsize.cumsum()])
            
    def get_samples(self,nsamples,istart=0,rand=False):    
        # If we are reading chain, it will be handled here 
        # istart -  will set row index to start getting the samples 
        
        if rand and not self.brange is None:
            ntot=self.method['samples'].shape[0]
            if nsamples>ntot:
                print('nsamples=%s, ntotal_chian=%s'%(nsamples,ntot))
                raise
            idx=np.random.randint(0,high=ntot,size=nsamples)
        else:
            idx=np.arange(istart,nsamples+istart)

        s,lnp,w=self.gd.arrays()            
                
        return s[idx,0:self.ndim],lnp[idx],w[idx]
        

    def evidence(self,verbose=None,rand=False,info=False,
                      profile=False,rprior=1,pos_lnp=False,
                      nproc=-1,prewhiten=True):
        #
        # MLE=maximum likelihood estimate of evidence:
        #
        
            
        if verbose is None:
            verbose=self.verbose
            
        kmax=self.kmax
        ndim=self.ndim
        
        MLE = np.zeros((self.nbatch,kmax))

        #get covariance matrix of chain
        #ChainCov=self.gd.samples.getCovMat()
        #eigenVal,eigenVec = np.linalg.eig(ChainCov)
        #Jacobian = math.sqrt(np.linalg.det(ChainCov))
        #ndim=len(eigenVal)
        
        # Loop over different numbers of MCMC samples (=S):
        itot=0
        for ipow,nsample in zip(self.idbatch,self.nchain):                
            S=int(nsample)            
            DkNN    = np.zeros((S,kmax))
            indices = np.zeros((S,kmax))
            volume  = np.zeros((S,kmax))
            
            samples_raw,logL,weight=self.get_samples(S,istart=itot,rand=rand)
            #
            if pos_lnp: logL=-logL
                
            # Renormalise loglikelihood (temporarily) to avoid underflows:
            logLmax = np.amax(logL)
            fs    = logL-logLmax
                        
            #print('(mean,min,max) of LogLikelihood: ',fs.mean(),fs.min(),fs.max())
            
            if prewhiten:
                # Covariance matrix of the samples, and eigenvalues (in w) and eigenvectors (in v):
                ChainCov = np.cov(samples_raw.T)
                eigenVal,eigenVec = np.linalg.eig(ChainCov)                
                Jacobian = math.sqrt(np.linalg.det(ChainCov))

                # Prewhiten:  First diagonalise:
                samples = np.dot(samples_raw,eigenVec);

                #print('EigenValues.shape,ndim',eigenVal.shape,ndim)
                #print('EigenValues=',eigenVal)
                # And renormalise new parameters to have unit covariance matrix:
                for i in range(ndim):
                    samples[:,i]= samples[:,i]/math.sqrt(eigenVal[i])
            else:
                #no diagonalisation
                Jacobian=1
                samples=samples_raw

            # Use sklearn nearest neightbour routine, which chooses the 'best' algorithm.
            # This is where the hard work is done:
            nbrs = NearestNeighbors(n_neighbors=kmax, 
                                    algorithm='auto',n_jobs=nproc).fit(samples)
            DkNN, indices = nbrs.kneighbors(samples)                
    
            # Create the posterior for 'a' from the distances (volumes) to nearest neighbour:
            for k in range(1,self.kmax):
                for j in range(0,S):        
                    # Use analytic formula for the volume of ndim-sphere:
                    volume[j,k] = math.pow(math.pi,ndim/2)*math.pow(DkNN[j,k],ndim)/sp.gamma(1+ndim/2)
                
                
                #print('volume minmax: ',volume[:,k].min(),volume[:,k].max())
                #print('weight minmax: ',weight.min(),weight.max())
                
                # dotp is the summation term in the notes:
                dotp = np.dot(volume[:,k]/weight[:],np.exp(fs))
        
                # The MAP value of 'a' is obtained analytically from the expression for the posterior:
                amax = dotp/(S*k+1.0)
    
                # Maximum likelihood estimator for the evidence
                SumW     = np.sum(weight)
                #print('SumW*S*amax*Jacobian',SumW,S,amax,Jacobian)
                MLE[ipow,k] = math.log(SumW*S*amax*Jacobian) + logLmax + math.log(rprior)
            
                # Output is: for each sample size (S), compute the evidence for kmax-1 different values of k.
                # Final columm gives the evidence in units of the analytic value.
                # The values for different k are clearly not independent. If ndim is large, k=1 does best.
                if self.brange is None:
                    #print('(mean,min,max) of LogLikelihood: ',fs.mean(),fs.min(),fs.max())
                    if verbose>1:
                        print('k={},nsample={}, dotp={}, median_volume={}, a_max={}, MLE={}'.format( 
                            k,S,dotp,statistics.median(volume[:,k]),amax,MLE[ipow,k]))
                
                else:
                    if verbose>1:
                        if ipow==0: 
                            print('(iter,mean,min,max) of LogLikelihood: ',ipow,fs.mean(),fs.min(),fs.max())
                            print('-------------------- useful intermediate parameter values ------- ')
                            print('nsample, dotp, median volume, amax, MLE')                
                        print(S,k,dotp,statistics.median(volume[:,k]),amax,MLE[ipow,k])

        #MLE[:,0] is zero - return only from k=1
        if self.brange is None:
            MLE=MLE[0,1:]
        else:
            MLE=MLE[:,1:]
            
        if verbose>0:
            print()
            print('MLE[k=(1,2,3,4)] = ',MLE)
            print()
        
        if info:
            return MLE, self.info
        else:  
            return MLE
    
           

#===============================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        method=sys.argv[1]
    else:
        print("")        
        print('        Usage: python MCEvidence.py <path/to/chain/file>')
        print("")
        print('        Optionaly the first argument can be a ')
        print('          file name to python class with "sampler" method')
        print("")
        sys.exit()

    if len(sys.argv) > 2:
        verbose=sys.argv[2]
    else:
        verbose=1
    
    print('Using Chain: ',method)
    mce=MCEvidence(method,verbose=verbose)
    mce.evidence()
