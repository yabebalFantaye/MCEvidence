'''
Planck MCMC chains evidence analysis. The data is available from [1].

Parameters
---------

Parallized version to compute evidence from Planck chains
We will analyze all schains in PLA folder

Returns
---------

The code writes results to terminal as well as a file. The default path
to the output files is

.. path:: planck_mce_fullGrid_R2/

Notes
---------

The full analysis using a single MPI process takes about ~30mins.


Examples
---------

To run the evidence estimation using 6 MPI processes

.. shell:: mpirun -np 6 python mce_pla.py

References
-----------

.. [1] Fullgrid Planck MCMC chains:
http://irsa.ipac.caltech.edu/data/Planck/release_2/ancillary-data/cosmoparams/COM_CosmoParams_fullGrid_R2.00.tar.gz


'''
from __future__ import absolute_import
from __future__ import print_function
import sys
import os
import glob
import numpy as np
import pandas as pd
from tabulate import tabulate
import pickle
import fileinput
from mpi4py import MPI
from argparse import ArgumentParser
import logging
#
from MCEvidence import MCEvidence


#---------------------------------------
#---- Extract command line arguments ---
#---------------------------------------
parser = ArgumentParser(description='Planck Chains MCEvidence.')

# Add options
parser.add_argument("-k", "--kmax",
                    dest="kmax",
                    default=2,
                    type=int,
                    help="scikit maximum K-NN ")
parser.add_argument("-nc", "--nchain",
                    dest="nchain",
                    default=0,
                    type=int,
                    help="How many chains to use (default=None - use all available) ")
parser.add_argument("-nd", "--ndata",
                    dest="ndata",
                    default=0,
                    type=int,                    
                    help="How many data cases to use (default=None - use all available) ")
parser.add_argument("-nm", "--nmodel",
                    dest="nmodel",
                    default=0,
                    type=int,                    
                    help="How many model cases to use (default=None - use all chains) ")
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
parser.add_argument("-o","--out", "--outdir",
                    dest="outdir",
                    default='planck_mce_fullGrid_R2',
                    help="Output directory name")
parser.add_argument("--N","--name",
                    dest="name",
                    default='mce',
                    help="base name for output files")
parser.add_argument("-v", "--verbose",
                    dest="verbose",
                    default=1,
                    type=int,
                    help="increase output verbosity")

args = parser.parse_args()

#-----------------------------
#------ control parameters----
#-----------------------------
kmax=args.kmax
nchain=args.nchain 
nmodel=args.nmodel
ndata=args.ndata
outdir=args.outdir
basename=args.name
burnfrac=args.burnfrac
thinfrac=args.thinfrac
verbose=args.verbose

#assert that kmax, the maximum kth nearest
#neighbour to use is >=2
assert kmax >= 2,'kmax must be >=2'

#---------------------------------------
#----- set basic logging
#---------------------------------------

if verbose==0:
    logging.basicConfig(level=logging.WARNING)
elif verbose==1:
    logging.basicConfig(level=logging.INFO)    
else:
    logging.basicConfig(level=logging.DEBUG)     

#
logger = logging.getLogger(__name__)


#-----------------------------
#-------- Initialize MPI -----
#-----------------------------
def mpi_load_balance(MpiSize,nload):
    nmpi_pp=np.zeros(MpiSize,dtype=np.int)
    nmpi_pp[:]=nload/MpiSize
    r=nload%MpiSize
    if r != 0:
        nmpi_pp[1:r-1]=nmpi_pp[1:r-1]+1

    return nmpi_pp

mpi_size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
comm = MPI.COMM_WORLD

#print all command line arguments passed
if rank==0:
    print(args)

## If parallel MPI-IO is to be used
#amode=MPI.MODE_WRONLY
#fhandle = MPI.File.Open(comm, fout, amode)

#---------------------------------------------------
#------- Path and sub-directory folders ------------
#---------------------------------------------------
rootdir='COM_CosmoParams_fullGrid_R2.00'

#list of cosmology parameters
cosmo_params=['omegabh2','omegach2','theta','tau','omegak','mnu','meffsterile','w','wa',
              'nnu','yhe','alpha1','deltazrei','Alens','Alensf','fdm','logA','ns','nrun',
              'nrunrun','r','nt','ntrun','Aphiphi']
    
# Types of model to consider. Below a more
# comprehensive list is defined using wildcards.
# The function avail_data_list() extracts all data names
# available in the planck fullgrid directory.
DataSets=['plikHM_TT_lowTEB','plikHM_TT_lowTEB_post_BAO','plikHM_TT_lowTEB_post_lensing','plikHM_TT_lowTEB_post_H070p6','plikHM_TT_lowTEB_post_JLA','plikHM_TT_lowTEB_post_zre6p5','plikHM_TT_lowTEB_post_BAO_H070p6_JLA','plikHM_TT_lowTEB_post_lensing_BAO_H070p6_JLA','plikHM_TT_lowTEB_BAO','plikHM_TT_lowTEB_BAO_post_lensing','plikHM_TT_lowTEB_BAO_post_H070p6','plikHM_TT_lowTEB_BAO_post_H070p6_JLA','plikHM_TT_lowTEB_lensing','plikHM_TT_lowTEB_lensing_post_BAO','plikHM_TT_lowTEB_lensing_post_zre6p5','plikHM_TT_lowTEB_lensing_post_BAO_H070p6_JLA','plikHM_TT_tau07plikHM_TT_lowTEB_lensing_BAO','plikHM_TT_lowTEB_lensing_BAO_post_H070p6','plikHM_TT_lowTEB_lensing_BAO_post_H070p6_JLA','plikHM_TTTEEE_lowTEB','plikHM_TTTEEE_lowTEB_post_BAO','plikHM_TTTEEE_lowTEB_post_lensing','plikHM_TTTEEE_lowTEB_post_H070p6','plikHM_TTTEEE_lowTEB_post_JLA','plikHM_TTTEEE_lowTEB_post_zre6p5','plikHM_TTTEEE_lowTEB_post_BAO_H070p6_JLA','plikHM_TTTEEE_lowTEB_post_lensing_BAO_H070p6_JLA','plikHM_TTTEEE_lowl_lensing','plikHM_TTTEEE_lowl_lensing_post_BAO','plikHM_TTTEEE_lowl_lensing_post_BAO_H070p6_JLA','plikHM_TTTEEE_lowTEB_lensing']


# Types of model to consider. Below a more
# comprehensive list is defined using wildcards.
# The function avail_model_list() extracts all data names
# available in the planck fullgrid directory.
Models={}
Models['model']=['base','base_omegak','base_Alens','base_Alensf','base_nnu','base_mnu',\
                 'base_nrun','base_r','base_w','base_alpha1','base_Aphiphi','base_yhe',\
                 'base_mnu_Alens','base_mnu_omegak','base_mnu_w','base_nnu_mnu',\
                 'base_nnu_r','base_nrun_r','base_nnu_yhe','base_w_wa',\
                 'base_nnu_meffsterile','base_nnu_meffsterile_r']

 
#---------------------------------------
#-------- define some useful functions -
#---------------------------------------
def avail_data_list(mm):
    '''
    Given model name, extract all available data names
    '''    
    l=glob.glob( '{0}/{1}/*/*_1.txt'.format(rootdir,mm) )
    l1=[x.split('_1')[0] for x in l]
    l2=[x.split('base_')[1] for x in l1]
    return l1,l2

def avail_model_list(dd,nmax=0,sorter=Models['model']):
    '''
    Given data name, extract all available models
    If sorter is not None, sorting will be based
    according to the order of sorter
    '''
    df=pd.DataFrame()
    l=glob.glob( '{0}/*/*/*_{1}_1.txt'.format(rootdir,dd) )
    df['l1']=[x.split('_1')[0] for x in l]    
    df['l2']=df['l1'].apply(lambda x:x.split('/')[1])
    
    #sort df based on sorter order
    if sorter:
        df['l2'] = df['l2'].astype("category")
        df['l2'].cat.set_categories(sorter, inplace=True)    
    df=df.sort_values('l2')

    if nmax>0:
        df=df.iloc[0:nmax]
    return df['l1'].values,df['l2'].values

def iscosmo_param(p,l=cosmo_params):
    '''
    check if parameter 'p' is cosmological or nuisance
    '''
    return p in l

def params_info(fname):
    '''
    Extract parameter names, ranges, and prior space volume
    from CosmoMC *.ranges file
    '''
    par=np.genfromtxt(fname+'.ranges',dtype=None,names=('name','min','max'))#,unpack=True)
    parName=par['name']
    parMin=par['min']
    parMax=par['max']
    
    parMC={'name':[],'min':[],'max':[],'range':[]}
    for p,pmin,pmax in zip(parName, parMin,parMax):
        if not np.isclose(pmax,pmin) and iscosmo_param(p):
            parMC['name'].append(p)
            parMC['min'].append(pmin)
            parMC['max'].append(pmax)
            parMC['range'].append(np.abs(pmax-pmin))
    #
    parMC['str']=','.join(parMC['name'])
    parMC['ndim']=len(parMC['name'])
    parMC['volume']=np.array(parMC['range']).prod()
    
    return parMC


#----------------------------------------------------------
#------- define which model, data etc. list to be used ----
#----------------------------------------------------------
if ndata>0:
    data_list=DataSets[0:ndata]
else:
    data_list=DataSets
    
model_list=Models['model'] 
logger.debug('len(data_list)={}, len(model_list)={}'.format(len(data_list),len(model_list)))


#------ mpi load balancing ---
main_loop_list=data_list
nload=len(main_loop_list)
lpp=mpi_load_balance(mpi_size,nload)


#--------------------------------------
#----Evidence calculation starts here -
#--------------------------------------

all_df={} #dictionary to store all results

if nchain == 0:  #use all available chains
    mce_cols=['AllChains']
    chains_extension_list=['']
    outdir='{}/AllChains/'.format(outdir) 
    outdir_data='{}/csv'.format(outdir)    
else:
    chains_extension_list=['_%s.txt'%x for x in range(1,nchain+1)]
    mce_cols=['chain%s'%k for k in range(1,nchain+1)]
    outdir='{}/SingleChains/'.format(outdir) 
    outdir_data='{}/csv'.format(outdir)        

fout_txt='%s/%s_{}.txt'%(outdir,basename)
fout_df='%s/%s_{}.csv'%(outdir_data,basename)
if not os.path.exists(outdir_data):
    os.makedirs(outdir_data) 

# Column names for useful information outputs
mce_info_cols=['PriorVol','ndim','N_read','N_used']

for ipp in range(lpp[rank]):  #loop over data
    ig=ipp+lpp[0:rank-1].sum()
    kk=main_loop_list[ig]
    logger.debug('*** mpi_rank, idd, loop_key',rank, ig, kk)

    kk_name='data'
    idd=ig
    dd=kk
    dd_dir=dd.split('_post_')[0]  
    dd_name=dd #dd.split('plikHM_')[0]    

    path_list, name=avail_model_list(dd,nmax=nmodel)
    mce=np.zeros((len(path_list),len(mce_cols)))
    mce_info={ k:[] for k in mce_info_cols }

    # prior volumes will be normalized by
    # the volume of the base model 
    vol_norm=1.0    
    for imm,mm,fname in zip(range(len(name)),name, path_list): #model loop
        if glob.glob(fname+'*.txt'):
            
            parMC=params_info(fname)
            if mm=='base':  #base model
                vol_norm=parMC['volume']
                
            prior_volume=parMC['volume']/vol_norm 
            ndim=parMC['ndim'] 
            #            
            mce_info['PriorVol'].append(prior_volume)
            mce_info['ndim'].append(ndim)            
            #
            logger.debug('***model: {},  ndim:{}, volume:{}, name={}'.format(mm,ndim,prior_volume,parMC['name']))
            #
            nc_read=''
            nc_use=''

            for icc, cext in enumerate(chains_extension_list):
                fchain=fname+cext
                e,info = MCEvidence(fchain,ndim=ndim,
                                    priorvolume=prior_volume,
                                    kmax=kmax,verbose=verbose,burnfrac=burnfrac,
                                    thinfrac=thinfrac).evidence(info=True,pos_lnp=False)
                mce[imm,icc]=e[0]
                icc+=1
                nc_read=nc_read+'%s,'%info['Nsamples_read']
                nc_use=nc_use+'%s,'%info['Nsamples']
                
            mce_info['N_read'].append(nc_read)
            mce_info['N_used'].append(nc_use)
        else:
            print('*** not available: ',fname)
            mce[imm,:]=np.nan
            mce_info['N_read'].append('')
            mce_info['N_used'].append('')
            mce_info['PriorVol'].append(0)
            mce_info['ndim'].append(0)             
    
    # At this stage evidence for a single data and all available
    # models is computed. Put the array in pandas DataFrame
    # and save it to a file 
    if not np.all(np.isnan(mce)):
        df = pd.DataFrame(mce,index=name,columns=mce_cols)
        df_mean=df.mean(axis=1)        
        if nchain>0:
            df_std=df.std(axis=1)        
            df['Mean_lnE_k1'] =df_mean
            df['Err_lnE_k1'] = df_std/np.sqrt(nchain*1.0)
        df['delta_lnE_k1'] =df_mean-df_mean.max()
        for k in mce_info_cols:
            df[k]=mce_info[k]

        #collect delta_lnE in a dictionary
        all_df[dd] = df['delta_lnE_k1']
        
        # print info    
        logging.info('--------------- {}---------'.format(kk))
        if verbose>0:
            print(tabulate(df, headers='keys', tablefmt='psql',floatfmt=".2f", numalign="left"))

        #--------- outputs ----------
        # first write to text file
        fout=fout_txt.format(kk)
        logging.info('rank={}, writing file to {}'.format(rank,fout))
        fhandle=open(fout, 'w')
        if rank==0:
            fhandle.write('\n')
            fhandle.write('############## RootDirectory={} ########\n'.format(rootdir))
            fhandle.write('\n')

        fhandle.write('\n')                
        fhandle.write('************ {} ************'.format(kk))
        fhandle.write('\n')                
        fhandle.write(tabulate(df, headers='keys', tablefmt='psql',floatfmt=".2f", numalign="left"))
        fhandle.write('\n')
        fhandle.close()

        # write dataframe to csv file
        fout=fout_df.format(kk)
        df.to_csv(fout)
   
        #--------- big MPI loop finish here ----

#--------------------------------
# wait for all process to finish
#--------------------------------
comm.Barrier() 


#----------------------------------------------------
#-- concatnate all output text files to a single file
#----------------------------------------------------
if rank==0:
    fmain='{}/mce_planck_fullgrid.txt'.format(outdir)
    fout_list=[fout_txt.format(kk) for kk in main_loop_list]
    print('all outputs being written to ',fmain)
    with open(fmain,'w') as outfile:        
        for fin in fout_list:
            if os.path.exists(fin):
                with open(fin) as inputfile:
                    for line in inputfile:
                        outfile.write(line)

    #delete all single files
    for fname in fout_list:
        if os.path.exists(fname):
            os.remove(fname)

#---------------------------------------
# gather all delta_lnE in one big array
#---------------------------------------
all_df = comm.gather(all_df, root=0)
if rank==0:
    logger.debug('after gather type(delta_lnE_df)=',type(all_df))
    all_df={ k: v for d in all_df for k, v in d.items() }
    if verbose>1:
        print('after_gather and concat: all_df.keys:',all_df.keys())

    # Save a dictionary into a pickle file.
    fout_pkl='{0}/delta_lnE_all_dict.pkl'.format(outdir_data)
    logger.info('writting : %s '%fout_pkl)
    pickle.dump(all_df, open(fout_pkl, "wb") )

    #concat all
    big_df=pd.DataFrame(index=model_list)
    for dd,df in all_df.items():
        big_df[dd]=df


    #sort big_df based on DataSets order
    df=big_df.T
    s = pd.Series(df.index.values,dtype='category')
    s.cat.set_categories(DataSets, inplace=True)
    big_df=df.reindex(s.sort_values()).T
        
    # Save a dictionary into a pickle file.
    fout_pkl='{0}/delta_lnE_all_df.pkl'.format(outdir_data)
    logger.info('writting : %s '%fout_pkl)    
    pickle.dump(big_df, open(fout_pkl, "wb") )
    
    # #read
    #big_df=pickle.load( open(fout_pkl, "rb" ) )

    #
    fout='{0}/delta_lnE_all.txt'.format(outdir)
    #logger.info('writting : %s '%fout)    
    fhandle=open(fout, 'w')    
    fhandle.write('\n')
    fhandle.write('############## RootDirectory={} ########\n'.format(rootdir))
    fhandle.write('\n')
    #print Long Column Names as header
    newName=[]
    for ik,k in enumerate(big_df.keys()):        
        nk='C%s'%ik
        fhandle.write('# {}={} \n'.format(nk,k))
        newName.append(nk) 
    big_df.columns=newName
    fhandle.write(tabulate(big_df, headers='keys', tablefmt='psql',floatfmt=".2f", numalign="left"))
    fhandle.write('\n')
    fhandle.close()
#---------------------------------------------
