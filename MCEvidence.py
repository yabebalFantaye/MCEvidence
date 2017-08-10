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
import glob
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
import logging
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__author__ = "Yabebal Fantaye"
__email__ = "yabi@aims.ac.za"
__license__ = "MIT"
__version__ = "0.1.1"
__status__ = "Development"

np.random.seed(1)

def weighted_thin(weights,thin_unit):
    '''
    Given a weight array, perform thinning.
    If the all weights are equal, this should 
    be equivalent to selecting every N/((1-thinfrac)*N)
    where N=len(weights).
    '''
    
    N=len(weights)
    if thin_unit==0: return range(N),weights
        
    if thin_unit<1:
        N2=np.int(N*(1.0-thin_unit))
    else:
        N2=N//thin_unit
    
    #bin the weight index to have the desired length
    #this defines the bin edges
    bins = np.linspace(-1, N, N2+1) 
    #this collects the indices of the weight array in each bin
    ind = np.digitize(np.arange(N), bins)  
    #this gets the maximum weight in each bin
    thin_ix=pd.Series(weights).groupby(ind).idxmax().tolist()
    thin_ix=np.array(thin_ix,dtype=np.intp)

    #print('ind=',ind[0:100])
    #print('bins=',np.array(bins[0:100],dtype=np.intp))
    #print('thin_ix=',thin_ix[0:100])
    #print('old_weight',weights[1:100])
    #print('new_weight',weights[thin_ix][0:99])
    #get the new weight by using weighted histogram
    #new_weight, _=np.histogram(np.arange(N), bins=N2, normed=False, weights=weights)    
    return {'ix':thin_ix, 'w':weights[thin_ix]}

def thin_indices(weights, factor):
    """
    Ref: 
    http://getdist.readthedocs.io/en/latest/_modules/getdist/chains.html#WeightedSamples.thin
    
    Indices to make single weight 1 samples. Assumes integer weights.

    :param factor: The factor to thin by, should be int.
    :param weights: The weights to thin, 
    :return: array of indices of samples to keep
    """
    numrows = len(weights)
    norm1 = np.sum(weights)
    weights = weights.astype(np.int)
    norm = np.sum(weights)

    if abs(norm - norm1) > 1e-4:
        print('Can only thin with integer weights')
        raise 
    if factor != int(factor):
        print('Thin factor must be integer')
        raise 
    factor = int(factor)
    if factor >= np.max(weights):
        cumsum = np.cumsum(weights) // factor
        # noinspection PyTupleAssignmentBalance
        _, thin_ix = np.unique(cumsum, return_index=True)
    else:
        tot = 0
        i = 0
        thin_ix = np.empty(norm // factor, dtype=np.int)
        ix = 0
        mult = weights[i]
        while i < numrows:
            if mult + tot < factor:
                tot += mult
                i += 1
                if i < numrows: mult = weights[i]
            else:
                thin_ix[ix] = i
                ix += 1
                if mult == factor - tot:
                    i += 1
                    if i < numrows: mult = weights[i]
                else:
                    mult -= (factor - tot)
                tot = 0

    return {'ix':thin_ix,'w':weights[thin_ix]}

#========== 

try:
    '''
    If getdist is installed, use that to reach chains.
    Otherwise, use the minimal chain reader class implemented below.
    '''    
    from getdist import MCSamples, chains
    from getdist import plots, IniFile
    import getdist as gd

    #raise 
    #====================================
    #      Getdist wrapper
    #====================================    
    class MCSamples(object):
        #Ref:
        #http://getdist.readthedocs.io/en/latest/plot_gallery.html

        def __init__(self,str_or_dict,trueval=None,debug=False,
                    names=None,labels=None,px='x',**kwargs):
            #Get the getdist MCSamples objects for the samples, specifying same parameter
            #names and labels; if not specified weights are assumed to all be unity

            if debug:
                logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)

            if isinstance(str_or_dict,str):
                
                fileroot=str_or_dict
                self.logger.info('string passed. Loading chain from '+fileroot)                
                self.load_from_file(fileroot,**kwargs)
                
            elif isinstance(str_or_dict,dict):
                d=str_or_dict
                self.logger.info('Chain is passed as dict: keys='+','.join(d.keys()))
                
                chain=d['samples']
                loglikes=d['loglikes']
                weights=d['weights'] if 'weights' in d.keys() else np.ones(len(loglikes))
                ndim=chain.shape[1]

                if names is None:
                    names = ["%s%s"%('p',i) for i in range(ndim)]
                if labels is None:
                    labels =  ["%s_%s"%(px,i) for i in range(ndim)]
                    
                self.names=names
                self.labels=labels
                self.trueval=trueval
                self.samples = gd.MCSamples(samples=chain,
                                            loglikes=loglikes,
                                            weights=weights,
                                            names = names,
                                            labels = labels)
                #Removes parameters that do not vary
                self.samples.deleteFixedParams()
                #Removes samples with zero weight
                #self.samples.filter(weights>0)
                
            else:
               self.logger.info('Passed first argument type is: ',type(str_or_dict))                
               self.logger.error('first argument to samples2getdist should be a string or dict.')
               raise

            # a copy of the weights that can be altered
            # independently to the original weights
            self.adjusted_weights=np.copy(self.samples.weights)
            #
            self.nparamMC=self.samples.paramNames.numNonDerived()

        def importance_sample(self,isfunc):
            #importance sample with external function
            negLogLikes=isfunc(self.samples.getParams())
            scale= 0#np.min(negLogLikes)            
            self.adjusted_weights *= np.exp(-(negLogLikes-scale))
            #self.adjusted_weights *= negLogLikes 

            #if self.samples.loglikes is not None:
            #    self.samples.loglikes += negLogLikes
            #self.samples.weights *= np.exp(-negLogLikes) 
            #self.samples._weightsChanged()
            #self.samples.reweightAddingLogLikes(negLogLikes)

        def get_shape(self):
            return self.samples.samples.shape

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
            idchain=kwargs.pop('idchain', 0)
            self.logger.info('mcsample: rootname={}, idchain={}'.format(rootname,idchain))
            self.samples=gd.loadMCSamples(rootname,**kwargs)#.makeSingle()
            if idchain>0:
                self.samples.samples=self.samples.getSeparateChains()[idchain-1].samples
                self.samples.loglikes=self.samples.getSeparateChains()[idchain-1].loglikes
                self.samples.weights=self.samples.getSeparateChains()[idchain-1].weights            
                
            # if rootname.split('.')[-1]=='txt':
            #     basename='_'.join(rootname.split('_')[:-1])+'.paramnames'
            #     print('loading parameter names from: ',basename)
            #     self.samples.setParamNames(basename)
                
        def thin(self,nminwin=1,nthin=0):
            if nthin < 0:
                ncorr=max(1,int(self.samples.getCorrelationLength(nminwin)))
                self.logger.info('Acutocorrelation Length: ncorr=%s'%ncorr)                
            else:
                ncorr=nthin

            try:
            #if True:
                norig=len(self.samples.weights)
                #get weighted thin
                try:
                    d = thin_indices(self.samples.weights,ncorr)
                except:
                    d = weighted_thin(self.samples.weights,ncorr)
                new_w=d['w']
                thin_ix=d['ix']
                #apply thinning
                #print('thin_ix',type(thin_ix),type(thin_ix[0]),len(thin_ix),thin_ix[0:10])
                self.samples.setSamples(self.samples.samples[thin_ix, :],
                                            self.samples.weights[thin_ix],
                                            self.samples.loglikes[thin_ix])
                #copy the new weight
                self.adjusted_weights=np.copy(self.samples.weights)
            
                #self.samples.thin(ncorr)
                nnew=len(self.samples.weights)
                logger.info('Thinning with thin length={} #old_chain={},#new_chain={}'.format(ncorr,norig,nnew))
            except:
            #else:
                self.logger.info('Thinning not possible. Weight must be interger to apply thinning.')

        def thin_poisson(self,thinfrac=0.1,nthin=None):
            '''Obsolete: This will be deleted in the future '''
            #try:
            w=self.samples.weights*(1.0-thinfrac)
            new_w=np.array([float(np.random.poisson(x)) for x in w])
            thin_ix=np.where(new_w>0)[0]
            logger.info('Thinning with thinfrac={}. new_nsamples={},old_nsamples={}'.format(thinfrac,len(thin_ix),len(w)))
            self.samples.setSamples(self.samples.samples[thin_ix, :],
                                        new_w[thin_ix],
                                    self.samples.loglikes[thin_ix]) #.makeSingle()
            self.adjusted_weights=np.copy(self.samples.weights)
            
            #except:
            #    self.logger.info('Poisson based thinning not possible.')

        def removeBurn(self,remove=0.2):
            self.samples.removeBurn(remove)
            
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
    
    '''
    getdist is not installed
    use a simple chain reader
    '''
    class MCSamples(object):

        def __init__(self,str_or_dict,trueval=None,debug=False,
                    names=None,labels=None,px='x',**kwargs):
            #Get the getdist MCSamples objects for the samples, specifying same parameter
            #names and labels; if not specified weights are assumed to all be unity

            if debug:
                logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)
                
            if isinstance(str_or_dict,str):                
                fileroot=str_or_dict
                self.logger.info('Loading chain from '+fileroot)                
                d = self.load_from_file(fileroot,**kwargs)                
            elif isinstance(str_or_dict,dict):                
                d=str_or_dict
            else:
               self.logger.info('Passed first argument type is: %s'%type(str_or_dict))                
               self.logger.error('first argument to samples2getdist should be a string or dict.')
               raise
           
            self.samples=d['samples']
            self.loglikes=d['loglikes']
            self.weights=d['weights'] if 'weights' in d.keys() else np.ones(len(self.loglikes))
            ndim=self.get_shape()[1]

            if names is None:
                names = ["%s%s"%('p',i) for i in range(ndim)]
            if labels is None:
                labels =  ["%s_%s"%(px,i) for i in range(ndim)]

            self.names=names
            self.labels=labels
            self.trueval=trueval
            self.nparamMC=self.get_shape()[1]

            # a copy of the weights that can be altered to
            # independently to the original weights
            self.adjusted_weights=np.copy(self.weights)
            
        def get_shape(self):            
            return self.samples.shape

        def importance_sample(self,isfunc):
            #importance sample with external function
            self.logger.info('Importance sampling ..')
            negLogLikes=isfunc(self.samples)
            scale=0 #negLogLikes.min()
            self.adjusted_weights *= np.exp(-(negLogLikes-scale))                     

        def load_from_file(self,fname,**kwargs):
            self.logger.warn('Loading file assuming CosmoMC columns order: weight loglike param1 param2 ...')
            try:
                DataTable=np.loadtxt(fname)
                self.logger.info(' loaded file: '+fname)
            except:
                d=[]
                idchain=kwargs.pop('idchain', 0)
                if idchain>0:
                    f=fname+'_{}.txt'.format(idchain)
                    DataTable=np.loadtxt(f)
                    self.logger.info(' loaded file: '+f)                    
                else:
                    self.logger.info(' loaded files: '+fname+'*')                    
                    for f in glob.glob(fname+'*'):
                        d.append(np.loadtxt(f))
                        
                    DataTable=np.concatenate(d)

            chain_dict={}
            #chain_dict['samples']=np.zeros((len(DataTable), ndim))
            chain_dict['weights']  =  DataTable[:,0]
            chain_dict['loglikes'] = DataTable[:,1]
            chain_dict['samples'] =  DataTable[:,2:]

            return chain_dict

        def thin(self,nthin=1):
            try:
                norig=len(self.weights)
                #call weighted thinning
                try:
                    #if weights are integers, use getdist algorithm
                    d = thin_indices(self.weights,ncorr)
                except:
                    #if weights are not integers, use internal algorithm
                    d = weighted_thin(self.weights,ncorr)
                    
                self.weights=d['w']
                thin_ix=d['ix']
                
                #now thin samples and related quantities
                self.samples=self.samples[thin_ix, :]
                self.loglikes=self.loglikes[thin_ix]

                self.adjusted_weights=self.weights.copy()
                
                nnew=len(self.weights)
                logger.info('Thinning with thin length={} #old_chain={},#new_chain={}'.format(nthin,norig,nnew))                
            except:
                self.logger.info('Thinning not possible.')
        
        def thin_poisson(self,thinfrac=0.1,nthin=None):
            w=self.weights*(1.0-thinfrac)
            new_w=np.array([float(np.random.poisson(x)) for x in w])
            thin_ix=np.where(new_w>0)[0]

            self.samples=self.samples[thin_ix, :]
            self.loglikes=self.loglikes[thin_ix]
            self.weights=new_w[thin_ix]
            self.adjusted_weights=self.weights.copy()
            
            logger.info('Thinning with thinfrac={}. new_nsamples={},old_nsamples={}'.format(thinfrac,len(thin_ix),len(w)))            
            #except:
            #    self.logger.info('Thinning not possible.')

        def removeBurn(self,remove=0):
            nstart=remove
            if remove<1:
                self.logger.info('burn-in: Removing {} % of the chain'.format(remove))                
                nstart=int(len(self.loglikes)*remove)
            else:
                self.logger.info('burn-in: Removing the first {} rows of the chain'.format(remove))
                
            self.samples=self.samples[nstart:, :]
            self.loglikes=self.loglikes[nstart:]
            self.weights=self.weights[nstart:]                

        def arrays(self):            
            s=self.samples
            lnp=-self.loglikes
            w=self.weights
            return s, lnp, w
           
#============================================================
#======  Here starts the main Evidence calculation code =====
#============================================================

class MCEvidence(object):
    def __init__(self,method,ischain=True,isfunc=None,
                     thinlen=0.0,burnlen=0.0,
                     ndim=None, kmax= 5, 
                     priorvolume=1,debug=False,
                     nsample=None,
                      nbatch=1,
                      brange=None,
                      bscale='',
                      verbose=1,args={},
                      **gdkwargs):
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
        
        :param gdkwargs (dict): arguments to be passed to getdist.

        :param verbose: chattiness of the run
        
        """
        #
        self.verbose=verbose
        if debug or verbose>1: logging.basicConfig(level=logging.DEBUG)
        if verbose==0: logging.basicConfig(level=logging.WARNING)            
        self.logger = logging.getLogger(__name__)
        
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
        self.priorvolume=priorvolume
        #
        self.ischain=ischain
        #
        self.fname=None
        #
        if ischain:
            
            if isinstance(method,str):
                self.fname=method      
                self.logger.debug('Using chains: ',method)
            else:
                self.logger.debug('dictionary of samples and loglike array passed')
                
        else: #python class which includes a method called sampler
            
            if nsample is None:
                self.nsample=100000
            else:
                self.nsample=nsample
            
            #given a class name, get an instance
            if isinstance(method,str):
                XClass = getattr(sys.modules[__name__], method)
            else:
                XClass=method
            
            if hasattr(XClass, '__class__'):
                self.logger.debug(__name__+': method is an instance of a class')
                self.method=XClass
            else:
                self.logger.debug(__name__+': method is class variable .. instantiating class')
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
        self.gd = MCSamples(method,debug=verbose>1,**gdkwargs)

        if burnlen>0:
            _=self.gd.removeBurn(remove=burnlen)
        if thinlen>0:
            self.logger.info('applying weighted thinning with thin length=%s'%thinlen)
            if thinlen>1:
                _=self.gd.thin(nthin=thinlen)
            else:
                _=self.gd.thin_poisson(thinfrac=thinlen)


        if isfunc:
            #try:
            self.gd.importance_sample(isfunc)
            #except:
            #    self.logger.warn('Importance sampling failed. Make sure getdist is installed.')
               
        self.info['NparamsMC']=self.gd.nparamMC
        self.info['Nsamples_read']=self.gd.get_shape()[0]
        self.info['Nparams_read']=self.gd.get_shape()[1]
        #

        #after burn-in and thinning
        self.nsample = self.gd.get_shape()[0]            
        if ndim is None: ndim=self.gd.nparamMC        
        self.ndim=ndim        
        #
        self.info['NparamsCosmo']=self.ndim
        self.info['Nsamples']=self.nsample
        #
        #self.info['MaxAutoCorrLen']=np.array([self.gd.samples.getCorrelationLength(j) for j in range(self.ndim)]).max()

        #print('***** ndim,nparamMC,MaxAutoCorrLen :',self.ndim,self.nparamMC,self.info['MaxAutoCorrLen'])
        
        #print('init minmax logl',method['lnprob'].min(),method['lnprob'].max())            
        self.logger.info('chain array dimensions: %s x %s ='%(self.nsample,self.ndim))
            
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
                self.logger.error('nbatch>1 but batch range is set to zero.')
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

        ntot=self.gd.get_shape()[0]
        
        if rand and not self.brange is None:
            if nsamples>ntot:
                self.logger.error('nsamples=%s, ntotal_chian=%s'%(nsamples,ntot))
                raise
            
            idx=np.random.randint(0,high=ntot,size=nsamples)
        else:
            idx=np.arange(istart,nsamples+istart)

        self.logger.info('requested nsamples=%s, ntotal_chian=%s'%(nsamples,ntot))
        s,lnp,w=self.gd.arrays()            
                
        return s[idx,0:self.ndim],lnp[idx],w[idx]
        

    def evidence(self,verbose=None,rand=False,info=False,
                      profile=False,pvolume=None,pos_lnp=False,
                      nproc=-1,prewhiten=True):
        '''

        MARGINAL LIKELIHOODS FROM MONTE CARLO MARKOV CHAINS algorithm described in Heavens et. al. (2017)
       
        Parameters
        ---------

        :param verbose - controls the amount of information outputted during run time
        :param rand - randomised sub sampling of the MCMC chains
        :param info - if True information about the analysis will be returd to the caller
        :param pvolume - prior volume
        :param pos_lnp - if input log likelihood is multiplied by negative or not
        :param nproc - determined how many processors the scikit package should use or not
        :param prewhiten  - if True chains will be normalised to have unit variance
        
        Returns
        ---------

        MLE - maximum likelihood estimate of evidence:
        self.info (optional) - returned if info=True. Contains useful information about the chain analysed
               

        Notes
        ---------

        The MCEvidence algorithm is implemented using scikit nearest neighbour code.


        Examples
        ---------

        To run the evidence estimation from an ipython terminal or notebook

        >> from MCEvidence import MCEvidence
        >> MLE = MCEvidence('/path/to/chain').evidence()
        

        To run MCEvidence from shell

        $ python MCEvidence.py </path/to/chain> 

        References
        -----------

        .. [1] Heavens etl. al. (2017)
        
        '''     
            
        if verbose is None:
            verbose=self.verbose

        #get prior volume
        if pvolume is None:
            logPriorVolume=math.log(self.priorvolume)
        else:
            logPriorVolume=math.log(pvolume)            

        self.logger.debug('log prior volume: ',logPriorVolume)
            
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

            samples_raw = np.zeros((S,ndim))
            samples_raw_cmc,logL,weight=self.get_samples(S,istart=itot,rand=rand)
            samples_raw[:,0:ndim] =  samples_raw_cmc[:,0:ndim]
            
            #We need the logarithm of the likelihood - not the negative log
            if pos_lnp: logL=-logL
                
            # Renormalise loglikelihood (temporarily) to avoid underflows:
            logLmax = np.amax(logL)
            fs    = logL-logLmax
                        
            #print('(mean,min,max) of LogLikelihood: ',fs.mean(),fs.min(),fs.max())
            
            if prewhiten:
                self.logger.info('Prewhitenning chains using sample covariance matrix ..')
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

            #print('samples, after prewhiten', samples[1000:1010,0:ndim])
            #print('Loglikes ',logLmax,logL[1000:1010],fs[1000:1010])
            #print('weights',weight[1000:1010])
            #print('EigenValues=',eigenVal)
            
            # Use sklearn nearest neightbour routine, which chooses the 'best' algorithm.
            # This is where the hard work is done:
            nbrs = NearestNeighbors(n_neighbors=kmax+1, 
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
                SumW     = np.sum(self.gd.adjusted_weights)
                self.logger.debug('********sumW={0},np.sum(Weight)={1}'.format(SumW,np.sum(weight)))
                MLE[ipow,k] = math.log(SumW*amax*Jacobian) + logLmax - logPriorVolume

                self.logger.debug('SumW={} \t S={} '.format(SumW,S))
                self.logger.debug('amax={} \t Jacobian={}'.format(amax,Jacobian))
                self.logger.debug('logLmax={} \t logPriorVolume={}'.format(logLmax,logPriorVolume))
                self.logger.debug('MLE={}:'.format(MLE[ipow,k]))
                print('---')
                # Output is: for each sample size (S), compute the evidence for kmax-1 different values of k.
                # Final columm gives the evidence in units of the analytic value.
                # The values for different k are clearly not independent. If ndim is large, k=1 does best.
                if self.brange is None:
                    #print('(mean,min,max) of LogLikelihood: ',fs.mean(),fs.min(),fs.max())
                    if verbose>1:
                        self.logger.info('k={},nsample={}, dotp={}, median_volume={}, a_max={}, MLE={}'.format( 
                            k,S,dotp,statistics.median(volume[:,k]),amax,MLE[ipow,k]))
                
                else:
                    if verbose>1:
                        if ipow==0: 
                            self.logger.info('(iter,mean,min,max) of LogLikelihood: ',ipow,fs.mean(),fs.min(),fs.max())
                            self.logger.info('-------------------- useful intermediate parameter values ------- ')
                            self.logger.info('nsample, dotp, median volume, amax, MLE')                
                        self.logger.info(S,k,dotp,statistics.median(volume[:,k]),amax,MLE[ipow,k])

        #MLE[:,0] is zero - return only from k=1
        if self.brange is None:
            MLE=MLE[0,1:]
        else:
            MLE=MLE[:,1:]

        if verbose>0:
            for k in range(1,self.kmax):
                print('   ln(B)[k={}] = {}'.format(k,MLE[k-1]))
            print('')
        if info:
            return MLE, self.info
        else:  
            return MLE
    
           
#===============================================
#===============================================


def iscosmo_param(p,cosmo_params=None):
    '''
    check if parameter 'p' is cosmological or nuisance
    '''
    if cosmo_params is None:
        #list of cosmology parameters
        cosmo_params=['omegabh2','omegach2','theta','tau','omegak','mnu','meffsterile','w','wa',
                      'nnu','yhe','alpha1','deltazrei','Alens','Alensf','fdm','logA','ns','nrun',
                      'nrunrun','r','nt','ntrun','Aphiphi']        
    return p in cosmo_params

def params_info(fname,cosmo=False):
    '''
    Extract parameter names, ranges, and prior space volume
    from CosmoMC *.ranges file
    '''
    logger.info('getting params info from %s.ranges'%fname)
    par=np.genfromtxt(fname+'.ranges',dtype=None,names=('name','min','max'))#,unpack=True)
    parName=par['name']
    parMin=par['min']
    parMax=par['max']
    
    parMC={'name':[],'min':[],'max':[],'range':[]}
    for p,pmin,pmax in zip(parName, parMin,parMax):
        #if parameter info is to be computed only for cosmological parameters
        pcond=iscosmo_param(p) if cosmo else True 
        #now get info
        if not np.isclose(pmax,pmin) and pcond:
            parMC['name'].append(p)
            parMC['min'].append(pmin)
            parMC['max'].append(pmax)
            parMC['range'].append(np.abs(pmax-pmin))
    #
    parMC['str']=','.join(parMC['name'])
    parMC['ndim']=len(parMC['name'])
    parMC['volume']=np.array(parMC['range']).prod()
    
    return parMC

#==============================================

if __name__ == '__main__':

    print('---')
    #---------------------------------------
    #---- Extract command line arguments ---
    #---------------------------------------
    parser = ArgumentParser(description='Planck Chains MCEvidence. Returns the log Bayesian Evidence computed using the kth NN')

    # positional args
    parser.add_argument("root_name",help='Root filename for MCMC chains or python class filename')
                        
    # optional args
    parser.add_argument("-k", "--kmax",
                        dest="kmax",
                        default=2,
                        type=int,
                        help="scikit maximum K-NN ")
    parser.add_argument("-ic", "--idchain",
                        dest="idchain",
                        default=0,
                        type=int,
                        help="Which chains to use - the id e.g 1 means read only *_1.txt (default=None - use all available) ")
    parser.add_argument("-np", "--ndim",
                        dest="ndim",
                        default=None,
                        type=int,                    
                        help="How many parameters to use (default=None - use all params) ")
    parser.add_argument("-b","--burnfrac", "--burnin","--remove",
                        dest="burnfrac",
                        default=0,
                        type=float,                    
                        help="Burn-in fraction")
    parser.add_argument("-t","--thin", "--thinfrac",
                        dest="thinfrac",
                        default=0,
                        type=float,
                        help="Thinning fraction")
    parser.add_argument("-v", "--verbose",
                        dest="verbose",
                        default=1,
                        type=int,
                        help="increase output verbosity")

    parser.add_argument('--cosmo', help='flag to compute prior_volume using cosmological parameters only',
                        action='store_true')

    args = parser.parse_args()

    #-----------------------------
    #------ control parameters----
    #-----------------------------
    method=args.root_name
    kmax=args.kmax
    idchain=args.idchain 
    ndim=args.ndim
    burnfrac=args.burnfrac
    thinfrac=args.thinfrac
    verbose=args.verbose
    
    if verbose>1: logging.basicConfig(level=logging.DEBUG)
    try:
        parMC=params_info(method,cosmo=args.verbose)
        if verbose>1: print(parMC)
        prior_volume=parMC['volume']
        logger.info('getting prior volume using cosmomc *.ranges output')
        logger.info('prior_volume=%s'%prior_volume)
    except:
        raise
        #print('setting prior_volume=1')
        #prior_volume=1
    print()
    print('Using file: ',method)    
    mce=MCEvidence(method,ndim=ndim,priorvolume=prior_volume,idchain=idchain,
                                    kmax=kmax,verbose=verbose,burnlen=burnfrac,
                                    thinlen=thinfrac)
    mce.evidence()

    print('* ln(B)[k] is the natural logarithm of the Baysian evidence estimated using the kth Nearest Neighbour.')
    print('')
