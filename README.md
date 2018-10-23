# MCEvidence
A python package implementing the MARGINAL LIKELIHOODS FROM MONTE CARLO MARKOV CHAINS algorithm described in Heavens et. al. (2017)

This code is tested in Python 2 version 2.7.12 and Python 3 version 3.5.2.

# Notes

The MCEvidence algorithm is implemented using scikit nearest neighbour code.

# Installation

To install this project into your machine using pip, do the following
                        
     $ git clone https://github.com/yabebalFantaye/MCEvidence
     $ cd MCEvidence
     $ pip install . --editable

The "--editable" or "-e" extension in the last command is to install the project in the editable mode.

To install this project with pip without clonning

     $ pip install git+https://github.com/yabebalFantaye/MCEvidence
     
# Examples
 
## To run the evidence estimation from an ipython terminal or notebook

    >> from MCEvidence import MCEvidence
    >> MLE = MCEvidence('/path/to/chain').evidence()
        
You can find a more advanced example that uses MCEvidence to analyse a set of MCMC chains in [planck_mcevidence.py](./planck_mcevidence.py). The result of our companion paper [No evidence for extensions to the standard cosmological model](http://adsabs.harvard.edu/abs/2017arXiv170403467H) is obtained using this code.

## To run MCEvidence from shell

    $ python MCEvidence.py </path/to/chain> [optional arguments]

You can check the allowed parameters by doing 
    $ python MCEvidence.py -h

The output is:

    usage: MCEvidence.py [-h] [-k KMAX] [-ic IDCHAIN] [-np NDIM] [-b BURNFRAC]
                         [-t THINFRAC] [-v VERBOSE] [--cosmo]
			                      root_name

    Planck Chains MCEvidence. Returns the log Bayesian Evidence computed using the
    kth NN.

     positional arguments:
       root_name             Root filename for MCMC chains or python class filename

     optional arguments:
       -h, --help            show this help message and exit
       -k KMAX, --kmax KMAX  scikit maximum K-NN
       -ic IDCHAIN, --idchain IDCHAIN
                        Which chains to use - the id e.g 1 means read only
                        *_1.txt (default=None - use all available)
       -np NDIM, --ndim NDIM
                        How many parameters to use (default=None - use all
                        params)             
       -b BURNFRAC, --burnfrac BURNFRAC, --burnin BURNFRAC, --remove BURNFRAC
                        Burn-in fraction
       -t THINFRAC, --thin THINFRAC, --thinfrac THINFRAC
                        Thinning fraction
       -vb VERBOSE, --verbose VERBOSE
                        Verbosity of the code while running: The mapping between verbose number
                        and the logging module levels are: 0: WARNNING, 1: INFO, 2: DEBUG
                        setting verbose>2 outputs EVERYTHING
       --paramsfile
                        text file name that contains additional parameter names to be consider as cosmological parameters
       --allparams              
              	        Flag to consider all parameters - prior_volume is computed using all parameters. By default only 
                        cosmological parameters listed in iscosmo_params function is considered.
                        For arbitrary parameter names, please modify the code accordingly.
                        If allparams is passed, consider using ndim<20 to get resonable evidence estimation.

       -np, --pvolume PRIORVOLUME
                        The prior volume to use if it can not be computed internally using cosmomc *.ranges or
                        montepython log.params files.

       --cross
                        Flag to split chain (s) into two sets to estimate cross Evidence. The DEFAULT is auto Evidence.
                        Cross EVIDENCE is computed using two independent chains. This means Nearest Neighbour of
                        a point  "A" in MCMC sample MC1 is searched in MCMC sample MC2.
                        THE ERROR ON THE EVIDENCE FROM (AUTO) EVIDENCE IS LARGER THAN THE CROSS EVIDENCE BY ~SQRT(2).
                        This is the result of : if the nearest neighbour of A is B, then the NN to B is LIKELY to be A.


# If you use the code, please cite the following paper

 .. [1] [Heavens et. al. (2017)](http://adsabs.harvard.edu/abs/2017arXiv170403472H)
