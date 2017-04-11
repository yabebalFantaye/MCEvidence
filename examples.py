'''
Collection of codes that can be used to test the MCEvidence code.
The examples below demostrate the validitiy of MCEvidence for 
three MCMC samplers:

* Gibbs Sampling
* PyStan NUT sampler
* EMCEE sampler

Two types of likelihood surface is considered

* Gaussian Linear Model - 3 dimensions
* N-dimensional Gaussian - 10 dimensions

'''

from __future__ import print_function
import IPython
import pickle
#
import os, sys, math,glob
import pandas as pd
import time
import numpy as np
import sklearn as skl
import statistics
from sklearn.neighbors import NearestNeighbors, DistanceMetric
import scipy.special as sp
#
from MCEvidence import MCEvidence


#pretty plots if seaborn is installed
try: 
    import seaborn as sns
    sns.set(style='ticks', palette='Set2',font_scale=1.5)
    #sns.set() 
except:
    pass


class glm_eg(object):
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
        self.theta_sample=reduce(np.dot, [np.linalg.inv(np.dot(self.D.T, self.D)), self.D.T, self.b])
        
    def quadratic(self,parameters):
        return parameters[0] + parameters[1]*self.x + parameters[2]*self.x**2

    def evidence(self):
        # Calculate the Bayesian Evidence               
        b=self.b
        D=self.D
        #
        num1 = np.log(det(2.0 * np.pi * np.linalg.inv(np.dot(D.T, D))))
        num2 = -0.5 * (np.dot(b.T, b) - reduce(np.dot, [b.T, D, np.linalg.inv(np.dot(D.T, D)), D.T, b]))
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
class gaussian_eg(object):
    def __init__(self,ndim=10,ndata=10000,verbose=1):
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
    
#====================================
#      PyStan chain example
#====================================
def glm_stan(iterations=10000,outdir='chains'):
    import pystan
    stanmodel='''
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
    glmq=glm_eg()
    df=pd.DataFrame()
    df['x1']=glmq.x
    df['x2']=glmq.x**2
    df['y']=glmq.y_sample

    data={'N':glmq.ndata,
               'K':glmq.ndim,
                'x':df[['x1','x2']],
               'y':glmq.y_sample}

   
    if os.path.exists(outdir):
        os.makedirs(outdir)
    cache_fname='{}/glm2d_pystan_chain.pkl'.format(outdir)
    #read chain from cache if possible 
    try:
        raise
        print('reading chain from: '+cache_fname)
        stan_chain = pickle.load(open(cache_fname, 'rb'))
    except:
        # Intialize pystan -- this will convert our pystan code into C++
        # and run MCMC
        fit = pystan.stan(model_code=stanmodel, data=data,
                          iter=1000, chains=4)

        # Extract PyStan chain for GLM example
        stan_chain=fit.extract(permuted=True)

        # Check input parameter recovery and estimate evidence
        if 'beta' in stan_chain.keys(): stan_chain['samples']=stan_chain.pop('beta')
        if 'lp__' in stan_chain.keys(): stan_chain['loglikes']=stan_chain.pop('lp__')
        
        
        print('writing chain in: '+cache_fname)
        with open(cache_fname, 'wb') as f:
                pickle.dump(stan_chain, f)


    theta_means = stan_chain['beta'].mean(axis=0)
    print('GLM example input parameter values: ',harry.theta)
    print('GLM example estimated parameter values: ',theta_means)


    # Here given pystan samples and log probability, we compute evidence ratio 
    mce=MCEvidence(stan_chain,verbose=2,ischain=True,brange=[3,4.2]).evidence()

    return mce

#====================================
#      Emcee chain example
#====================================
import emcee
class make_emcee_chain(object):
    # A wrapper to the emcee MCMC sampler
    #
    def __init__(self,model,nwalkers=500,nburn=300,arg={}):

        #check if model is string or not
        if isinstance(model,str):
            print('name of model: ',model)
            XClass = getattr(sys.modules[__name__], model)
        else:            
            XClass=model        

        #check if XClass is instance or not
        if hasattr(XClass, '__class__'): 
            print('instance of a model class is passed')
            self.model=XClass #it is instance 
        else:
            print('class variable is passed .. instantiating class')
            self.model=XClass(*arg)

        self.ndim=self.model.ndim

        #init emcee sampler
        self.nwalkers=nwalkers
        self.emcee_sampler = emcee.EnsembleSampler(self.nwalkers, 
                                             self.model.ndim, 
                                             self.model.lnprob)   

        # burnin phase
        pos0=self.model.pos(self.nwalkers)
        pos, prob, state  = self.emcee_sampler.run_mcmc(pos0, nburn)

        #save emcee state
        self.prob=prob
        self.pos=pos
        self.state=state

        #discard burnin chain 
        self.samples = self.emcee_sampler.flatchain        
        self.emcee_sampler.reset()

    def mcmc(self,nmcmc=2000,**kwargs):
        # perform MCMC - no resetting 
        # size of the chain increases in time
        time0 = time.time()
        #
        #pos=None makes the chain start from previous state of sampler
        self.pos, self.prob, self.state  = self.emcee_sampler.run_mcmc(self.pos,nmcmc,**kwargs)
        self.samples = self.emcee_sampler.flatchain    
        self.lnp = self.emcee_sampler.flatlnprobability
        #
        time1=time.time()
        #
        print('emcee total time spent: ',time1-time0)        
        print('samples shape: ',self.samples.shape)  

        return self.samples,self.lnp

    def Sampler(self,nsamples=2000):
        # perform MCMC and return exactly nsamples
        # reset sampler so that chains don't grow
        #
        N=(nsamples+self.nwalkers-1)/self.nwalkers #ceil to next integer
        print('emcee: nsamples, nmcmc: ',nsamples,N*self.nwalkers)
        #
        #pos=None makes the chain start from previous state of sampler
        self.pos, self.prob, self.state  = self.emcee_sampler.run_mcmc(self.pos,N)
        self.samples = self.emcee_sampler.flatchain    
        self.lnp = self.emcee_sampler.flatlnprobability
        self.emcee_sampler.reset()

        return self.samples[0:nsamples,:],self.lnp[0:nsamples]

    def vis(self,chain=None,figsize=(10,10),**kwargs):
        # Visualize the chains

        if chain is None:
            chain=self.samples

        fig = corner.corner(chain, labels=self.model.label, 
                                   truths=self.model.p,
                                   **kwargs)            

        fig.set_size_inches(figsize)  

    def info(self):
        print("Example using emcee sampling") 
        print('nwalkers=',self.walkers)
        try:
            self.model.info()
        except:
            pass
        print()  

def gaussian_emcee(nwalkers=300,thin=5,nmcmc=5000):
    #Evidence calculation based on emcee sampling
    mNd=gaussian_eg()
    mecNd=make_emcee_chain(mNd,nwalkers=nwalkers)
    samples,lnp=mecNd.mcmc(nmcmc=nmcmc,thin=thin)


    #estimate evidence
    chain={'samples':samples,'loglikes':lnp}
                          
    mce=MCEvidence(chain, verbose=2,ischain=True,                       
                       brange=[3,4.2]).evidence(rand=True)

    return mce

#===============================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        method=sys.argv[1]
    else:
        method='gaussian_eg'

    if len(sys.argv) > 2:
        nsamples=sys.argv[2]
    else:
        nsamples=10000

    if method in ['gaussian_eg','glm_eg']:
        print('Using example: ',method)

        #get class instance
        XClass = getattr(sys.modules[__name__], method)
        
        # Now Generate samples.
        print('Calling sampler to get MCMC chain: nsamples=',nsamples)
        samples,logl=XClass(verbose=2).Sampler(nsamples=nsamples)

        print('samples and loglikes shape: ',samples.shape,logl.shape)
        
        chain={'samples':samples,'loglikes':logl}       
        mce=MCEvidence(chain,thinlen=2,burnlen=0.1,verbose=2,ischain=True).evidence()
        
    else:
        mce=eval(method+'()')

