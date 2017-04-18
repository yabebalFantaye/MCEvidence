# MCEvidence
A python package implementing the MARGINAL LIKELIHOODS FROM MONTE CARLO MARKOV CHAINS algorithm described in Heavens et. al. (2017)

# Notes

The MCEvidence algorithm is implemented using scikit nearest neighbour code.


# Examples
 
To run the evidence estimation from an ipython terminal or notebook

    >> from MCEvidence import MCEvidence
    >> MLE = MCEvidence('/path/to/chain').evidence()
        

To run MCEvidence from shell

    $ python MCEvidence.py </path/to/chain> 

# References

 .. [1] Heavens etl. al. (2017)
