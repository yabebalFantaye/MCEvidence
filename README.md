# MCEvidence
A python package implementing the MARGINAL LIKELIHOODS FROM MONTE CARLO MARKOV CHAINS algorithm described in Heavens et. al. (2017)

# Notes

The MCEvidence algorithm is implemented using scikit nearest neighbour code.


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
       -v VERBOSE, --verbose VERBOSE
                        increase output verbosity
       --cosmo              
              	        Flag to compute prior_volume using cosmological
                        parameters only 

# If you use the code, please cite the following paper

 .. [1] [Heavens et. al. (2017)](http://adsabs.harvard.edu/abs/2017arXiv170403472H)
