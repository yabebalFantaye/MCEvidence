
# coding: utf-8

from __future__ import print_function
import IPython
import pickle
#
import os, sys, math
import pandas as pd
import time
import numpy as np
#import pandas as pd
import sklearn as skl
import statistics
from sklearn.neighbors import NearestNeighbors, DistanceMetric
#from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import scipy.special as sp
#
import eknn
from wrap import *

#pretty plots if seaborn is installed
try: 
    import seaborn as sns
    sns.set(style='ticks', palette='Set2',font_scale=1.5)
    #sns.set() 
except:
    pass


class harry_eg(object):
    def __init__(self,x=None,theta=None,
                 rms=0.2,ptheta=None,verbose=1):
        
        # Generate Data for a Quadratic Function
        if x is None:
            xmin        = 0.0
            xmax        = 4.0
            nDataPoints = 200
            x = np.linspace(xmin, xmax, nDataPoints)
        #data points
        self.x=x
        self.ndata=len(x)
        
        # Data simulation inputs
        if theta is None:
            theta0_true = 1.0
            theta1_true = 4.0
            theta2_true = -1.0
            theta = np.array([theta0_true, theta1_true, theta2_true])
        #parameters
        self.theta=theta
        self.ndim=len(theta)
        
        #flat priors on parameters 
        if ptheta is None:
            ptheta = np.repeat(10.0,self.ndim)

        # Generate quadratic data with noise
        self.y          = self.quadratic(self.theta)
        self.noise_rms = np.ones(self.ndata)*rms
        self.y_sample     = self.y + np.random.normal(0.0, self.noise_rms) 
        
        self.D      = np.zeros(shape = (self.ndata, self.ndim))
        self.D[:,0] = 1.0/self.noise_rms
        self.D[:,1] = self.x/self.noise_rms
        self.D[:,2] = self.x**2/self.noise_rms
        self.b      = self.y_sample/self.noise_rms               
        
        #Initial point to start sampling 
        self.theta_sample=reduce(np.dot, [inv(np.dot(self.D.T, self.D)), self.D.T, self.b])
        
    def quadratic(self,parameters):
        return parameters[0] + parameters[1]*self.x + parameters[2]*self.x**2

    def evidence(self):
        # Calculate the Bayesian Evidence               
        b=self.b
        D=self.D
        #
        num1 = np.log(det(2.0 * np.pi * inv(np.dot(D.T, D))))
        num2 = -0.5 * (np.dot(b.T, b) - reduce(np.dot, [b.T, D, inv(np.dot(D.T, D)), D.T, b]))
        den1 = np.log(self.ptheta.prod()) #prior volume
        #
        log_Evidence = num1 + num2 - den1 #(We have ignored k)
        #
        print('\nThe log-Bayesian Evidence is equal to: {}'.format(log_Evidence))
        
        return log_Evidence
        
    
    def gibbs_dist(self, params, label):
        # The conditional distributions for each parameter
        # This will be used in the Gibbs sampling 
        
        b=self.b
        D=self.D
        sigmaNoise=self.noise_rms
        x=self.x
        ndata=self.ndata
        
        #
        D0 = np.zeros(shape = (ndata, 2)); D0[:,0] = x/sigmaNoise; D0[:,1] = x**2/sigmaNoise 
        D1 = np.zeros(shape = (ndata, 2)); D1[:,0] = 1./sigmaNoise; D1[:,1] = x**2/sigmaNoise 
        D2 = np.zeros(shape = (ndata, 2)); D2[:,0] = 1./sigmaNoise; D2[:,1] = x/sigmaNoise 

        if label == 't0':
            theta_r = np.array([params[1], params[2]])
            v       = 1.0/sigmaNoise
            A       = np.dot(v.T, v)
            B       = -2.0 * (np.dot(b.T, v) - reduce(np.dot, [theta_r.T, D0.T, v]))
            mu      = -B/(2.0 * A)
            sig     = np.sqrt(1.0/A)

        if label == 't1':
            theta_r = np.array([params[0], params[2]])
            v       = x/sigmaNoise
            A       = np.dot(v.T, v)
            B       = -2.0 * (np.dot(b.T, v) - reduce(np.dot, [theta_r.T, D1.T, v]))
            mu      = -B/(2.0 * A)
            sig     = np.sqrt(1.0/A)

        if label == 't2':
            theta_r = np.array([params[0], params[1]])
            v       = x**2/sigmaNoise
            A       = np.dot(v.T, v)
            B       = -2.0 * (np.dot(b.T, v) - reduce(np.dot, [theta_r.T, D2.T, v]))
            mu      = -B/(2.0 * A)
            sig     = np.sqrt(1.0/A)

        return np.random.normal(mu, sig)

    def Sampler(self,nsamples=1000):

        b=self.b
        D=self.D
        
        Niters        = int(nsamples)
        trace         = np.zeros(shape = (Niters, 3))
        logLikelihood = np.zeros(Niters) 

        #previous state
        params=self.theta_sample
        
        for i in range(Niters):
            params[0]  = self.gibbs_dist(params, 't0')
            params[1]  = self.gibbs_dist(params, 't1')
            params[2]  = self.gibbs_dist(params, 't2')
        
            trace[i,:] = params

            logLikelihood[i] = -0.5 * np.dot((b - np.dot(D,trace[i,:])).T, (b - np.dot(D,trace[i,:]))) 

        #save the current state back to theta_sample
        self.theta_sample=params
        
        return trace, logLikelihood 
    
    def info(self):
        return '''Example adabted from Harry's Jupyter notebook. 
        \n{0}-dimensional Polynomial function.'''.format(self.ndim)    

    
#===================================
#   2d likelihood for emcee sampler
#==================================
# Define the posterior PDF
# Reminder: post_pdf(theta, data) = likelihood(data, theta) * prior_pdf(theta)
# We take the logarithm since emcee needs it.
#---------------    
class model_2d(object):
    def __init__(self,p=[-0.9594,4.294],pprior=None,
                 N=50,x=None,**kwargs):
    
        f=lambda t,s: np.array([t-s*abs(t),t+s*abs(t)])
        
        if pprior is None:
            self.pprior={'p'+str(i) : f(t,10) for i,t in enumerate(p) }
            
        self.label=self.pprior.keys()
        self.ndim=len(p)
        self.p=p        
        if x is None:
            self.N=N
            self.x = np.sort(10*np.random.rand(N))
        else:
            self.N=len(x)
            self.x=x
            
        self.y,self.yerr=self.data(**kwargs)
        
    # As prior, we assume an 'uniform' prior (i.e. constant prob. density)
    def inprior(self,t,i):
        prange=self.pprior[self.label[i]]
        if  prange[0] < t < prange[1]:
            return 1.0
        else:
            return 0.0

    def lnprior(self,theta):
        for i,t in enumerate(theta):
            if self.inprior(t,i)==1.0:
                pass
            else:
                return -np.inf
        return 0.0
        

    # As likelihood, we assume the chi-square. 
    def lnlike(self,theta):
        m, b = theta
        model = m * self.x + b
        return -0.5*(np.sum( ((self.y-model)/self.yerr)**2. ))

    def lnprob(self,theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta)

    def data(self,sigma=0.5,aerr=0.2):
        # Generate synthetic data from a model.
        # For simplicity, let us assume a LINEAR model y = m*x + b
        # where we want to fit m and b      
        yerr = aerr + sigma*np.random.rand(self.N)
        y = self.p[0]*self.x + self.p[1]
        y += sigma * np.random.randn(self.N) 
        return y,yerr
    
    def pos(self,nwalkers):
        # uniform sample over prior space
        # will be used as starting place for
        # emcee sampler
        r=np.random.rand(nwalkers,self.ndim)
        pos=r
        for i,k in enumerate(self.pprior):
            prange=self.pprior[k]            
            psize = prange.max() - prange.min()
            pos[:,i]=prange.min()+psize*r[:,i]
        return pos
    
    def vis(self,n=300,figsize=(10,10),**kwargs):
        # Visualize the chains
        try:
            import corner 
            fig = corner.corner(self.pos(n), 
                                labels=self.label, 
                                truths=self.p,**kwargs)            
            fig.set_size_inches(figsize) 
        except:
            print('corner package not installed - no plot is produced.')
            pass

#     
#============================================
class alan_eg(object):
    def __init__(self,ndim=10,ndata=100000,verbose=1):
        #  Generate data

        # Number of dimensions: up to 15 this seems to work OK. 
        self.ndim=ndim

        # Number of data points (not actually very important)
        self.ndata=ndata

        # Some fairly arbitrary mean values for the data.  
        # Standard deviation is unity in all parameter directions.
        std = 1.0
        self.mean  = np.zeros(ndim)
        for i in range(0,ndim):
            self.mean[i]  = np.float(i+1)
              
        # Generate random data all at once:
        self.d2d=np.random.normal(self.mean,std,size=(ndata,ndim))

        # Compute the sample mean and standard deviations, for each dimension
        # The s.d. should be ~1/sqrt(ndata))
        self.mean_sample = np.mean(self.d2d,axis=0)
        self.var_sample  = np.var(self.d2d,axis=0)
        #1sigma error on the mean values estimated from ndata points 
        self.sigma_mean  = np.std(self.d2d,axis=0)/np.sqrt(np.float(ndata))
            
        if verbose>0:
            std_sample  = np.sqrt(self.var_sample)
            print()
            print('mean_sample=',self.mean_sample) 
            print('std_sample=',std_sample)
            print()

    # Compute ln(likelihood)
    def lnprob(self,theta):      
        dM=(theta-self.mean_sample)/self.sigma_mean        
        return (-0.5*np.dot(dM,dM) -
                     self.ndim*0.5*np.log(2.0*math.pi) -
                     np.sum(np.log(self.sigma_mean)))
            
    # Define a routine to generate samples in parameter space:
    def Sampler(self,nsamples=1000):

        # Number of samples:                 nsamples
        # Dimensionality of parameter space: ndim
        # Means:                             mean
        # Standard deviations:               stdev

        
        ndim=self.ndim
        ndata=self.ndata
        mean=self.mean_sample
        sigma=self.sigma_mean
        #
        #Initialize vectors:
        theta = np.zeros((nsamples,ndim))
        f     = np.zeros(nsamples)

        # Generate samples from an ndim-dimension multivariate gaussian:
        theta = np.random.normal(mean,sigma,size=(nsamples,ndim))

        for i in range(nsamples):
            f[i]=self.lnprob(theta[i,:])

        return theta, f   
    def pos(self,n):
        # Generate samples over prior space volume
        return np.random.normal(self.mean_sample,5*self.sigma_mean,size=(n,self.ndim))
    
    def info(self):
        print("Example adabted from Alan's Jupyter notebook") 
        print('{0}-dimensional Multidimensional gaussian.'.format(self.ndim))
        print('ndata=',self.ndata)
        print()
    
#================================

import bayesglm as sglm
import pystan

harry_stanmodel='''
 data {
         int<lower=1> K;
         int<lower=0> N;
         real y[N];
         matrix[N,K] x;
 }
 parameters {
         vector[K] beta;
         real sigma;
 }
 model {         
         real mu[N];
         vector[N] eta   ;
         eta <- x*beta;
         for (i in 1:N) {
            mu[i] <- (eta[i]);
         };
         increment_log_prob(normal_log(y,mu,sigma));

 }
 '''   
harry=eknn.harry_eg()
df=pd.DataFrame()
df['x1']=harry.x
df['x2']=harry.x**2
df['y']=harry.y_sample

harry_data={'N':harry.ndata,
           'K':harry.ndim,
            'x':df[['x1','x2']],
           'y':harry.y_sample}
# Intialize pystan -- this will convert our pystan code into C++
# and run MCMC
#harry_fit = pystan.stan(model_code=harry_stanmodel, data=harry_data,
#                  iter=1000, chains=4)


# In[23]:

iterations=10000
class jeffry_prior():
    def __init__(self, sigma):
        self.sigma = sigma

    def __repr__(self):
        return self.to_string()

    def to_string(self):
        return "normal(0,{0})".format(self.sigma)
class normal_prior():
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return self.to_string()

    def to_string(self):
        return "normal({0},{1})".format(self.mu, self.sigma)


#priors=(((i+1,), jeffry_prior(np.sqrt(harry.ndata))) for i in range(harry.ndim-1) )

priors={"x%s"%(i+1) : normal_prior(harry.theta[i+1],0.2) for i in range(harry.ndim-1) }

cache_fn='chains/harry_pystan_chain.pkl'
#read chain from cache if possible 
try:
    raise
    print('reading chain from: '+cache_fn)
    harry_stan_chain = pickle.load(open(cache_fn, 'rb'))
except:
    harry_fit = sglm.stan_glm("y ~ x1 + x2", df, 
                              family=sglm.family.gaussian(), 
                              iterations=iterations) #,priors=priors

    # Extract PyStan chain for Harry's GLM example
    harry_stan_chain=harry_fit.extract(permuted=True)   
    print('writing chain in: '+cache_fn)
    with open(cache_fn, 'wb') as f:
            pickle.dump(harry_stan_chain, f)

    #print stan model
    harry_model=harry_fit.stanmodel.model_code.expandtabs() #.rsplit('\n') 
    with open('harry.stan', 'w') as f:   
        f.write(harry_model[:])

theta_means = harry_stan_chain['beta'].mean(axis=0)
print('estimated: ',theta_means)
print('input: ',harry.theta)

#np.testing.assert_allclose(theta_means, harry.theta, atol=.01)


# In[27]:

plt.plot(harry_stan_chain['lnprob'])
harry_stan_chain['lnprob']=harry_stan_chain['lnprob']+2*np.log(0.1/np.sqrt(1.0*harry.ndata))
plt.plot(harry_stan_chain['lnprob'])


# In[24]:

# Check input parameter recovery and estimate evidence
if 'beta' in harry_stan_chain.keys(): harry_stan_chain['samples']=harry_stan_chain.pop('beta')
if 'lp__' in harry_stan_chain.keys(): harry_stan_chain['lnprob']=harry_stan_chain.pop('lp__')
print(harry_stan_chain['samples'].shape)

#
gdstans=samples2gdist(harry_stan_chain['samples'],harry_stan_chain['lnprob'],
                     trueval=harry.theta,px='\\theta')
gdstans.corner(figsize=(10,10))
#gdstans.labels


# In[28]:

# Here given pystan samples and log probability, we compute evidence ratio 
eharry=eknn.echain(method=harry_stan_chain,verbose=2,ischain=True)
MLE=eharry.chains2evidence() 
eharry.vis_mle(MLE)


# In[ ]:




# ## Emcee example

# In[164]:

##learn about emcee sampler using help
#help(mec2d.sampler)


# ## emcee sampling using N-dimensional Gaussian likelihood

# In[210]:

#
#gd_mc.samples.getName()


# In[21]:

#Evidence calculation based on emcee sampling
mNd=eknn.alan_eg()
mecNd=make_emcee_chain(mNd,nwalkers=300)
samples,lnp=mecNd.mcmc(nmcmc=50000,thin=50)


# In[26]:

#corner plot can be done also using getdist wrapper
#getdist wrapper has a lot more functionality than just plotting
gd_mc=samples2gdist(samples,lnp,trueval=mNd.mean,px='m')
print('correlation length:',gd_mc.samples.getCorrelationLength(3))
gd_mc.samples.thin(20)
##gd_mc.corner()
#mecNd.emcee_sampler.get_autocorr_time(fast=True)


# In[28]:

thin_samples=gd_mc.samples.samples
thin_lnp=gd_mc.samples.loglikes

print(len(thin_lnp),thin_samples.shape)

#estimate evidence
ealan=eknn.echain(method={'samples':thin_samples,'lnprob':thin_lnp},
                  verbose=2,ischain=True,brange=[3,4.2])
MLE=ealan.chains2evidence(rand=True) 


# In[29]:

ealan.vis_mle(MLE)


# ## Emcee 2D example

# In[ ]:

#test model class .. visualise uniform sampling
m2d=eknn.model_2d()

#test emcee wrapper 
mec2d=make_emcee_chain(m2d,nwalkers=200)
chain2d,fs=mec2d.mcmc(nmcmc=500)


#let's trangle plot chain samples 
fig = corner.corner(chain2d, labels=["$m$", "$b$"], extents=[[-1.1, -0.8], [3.5, 5.]],
                      truths=m2d.p, quantiles=[0.16, 0.5, 0.84], 
                    show_titles=True, labels_args={"fontsize": 40})
fig.set_size_inches(10,10)

# Plot back the results in the space of data
#fig = plt.figure()
#xl = np.array([0, 10])
#for m, b in chain2d[np.random.randint(len(chain2d), size=100)]:
#    if m<0:
#        plt.plot(xl, m*xl+b, color="k", alpha=0.1)
    
#plt.plot(xl, m2d.p[0]*xl+m2d.p[1], color="r", lw=2, alpha=0.8)
#plt.errorbar(m2d.x, m2d.y, yerr=m2d.yerr, fmt=".k")
#plt.title('Input Data vs Samples (grey)')
#fig.set_size_inches(12, 8)


# In[ ]:




# In[ ]:



