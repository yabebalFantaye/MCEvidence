'''

Parallized version to compute evidence from Planck chains
We will analyze all schains in PLA folder


example:

    mpirun -np 6 python mce_pla.py

'''
from __future__ import absolute_import
from __future__ import print_function
import sys
import os
import glob
import numpy as np
import pandas as pd
import astropy as ap
from tabulate import tabulate
import pickle
from MCEvidence import MCEvidence

#--------------

if len(sys.argv) > 1:
    kmax=int(sys.argv[1])
else:
    kmax=5

print('kmax=',kmax,kmax>=2)
assert isinstance(kmax,int),'kmax must be int'
assert kmax >= 2,'kmax must be >=2'

if len(sys.argv) > 2:
    verbose=sys.argv[2]
else:
    verbose=0


rootdir='COM_CosmoParams_fullGrid_R2.00'
outdir='planck_mce_fullGrid_R2/'
if not os.path.exists(outdir):
    os.makedirs(outdir)
    
#-----------------------------
'''
Using bash lines to extract datasets from the fullgrid folder
Bash Command:
   ls COM_CosmoParams_fullGrid_R2.00/base/*/!(*post*) | grep _1.txt | grep -Po '(?<=base/).*(?=/base)' | xargs -I {} echo "{}",
'''

DataSets='''
lensonly
lensonly_BAO
lensonly_BAO_theta
lensonly_theta
plikDS_TT_lowTEB
plikHM_EE
plikHM_EE_lensing
plikHM_EE_lowEB
plikHM_EE_lowTEB
plikHM_TE
plikHM_TE_lensing
plikHM_TE_lowEB
plikHM_TE_lowTEB
plikHM_TT_lowEB
plikHM_TT_lowl
plikHM_TT_lowl_lensing
plikHM_TT_lowl_reion
plikHM_TT_lowTEB
plikHM_TT_lowTEB_lensing
plikHM_TT_tau07
plikHM_TTTEEE_lowEB
plikHM_TTTEEE_lowl
plikHM_TTTEEE_lowl_lensing
plikHM_TTTEEE_lowl_reion
plikHM_TTTEEE_lowTEB
plikHM_TTTEEE_lowTEB_lensing
plikHM_TTTEEE_tau07
plikHM_TT_WMAPTEB
WLonlyHeymans_BAO
WLonlyHeymans_BAO_theta
WLonlyHeymans
WLonlyHeymans_H070p6_BAO_theta
WLonlyHeymans_H070p6_theta
WMAP'''.split('\n')[1:]

DataSets=['plikHM_TT_lowTEB','plikHM_TT_lowTEB_post_BAO','plikHM_TT_lowTEB_post_lensing','plikHM_TT_lowTEB_post_H070p6','plikHM_TT_lowTEB_post_JLA','plikHM_TT_lowTEB_post_zre6p5','plikHM_TT_lowTEB_post_BAO_H070p6_JLA','plikHM_TT_lowTEB_post_lensing_BAO_H070p6_JLA','plikHM_TT_lowTEB_BAO','plikHM_TT_lowTEB_BAO_post_lensing','plikHM_TT_lowTEB_BAO_post_H070p6','plikHM_TT_lowTEB_BAO_post_H070p6_JLA','plikHM_TT_lowTEB_lensing','plikHM_TT_lowTEB_lensing_post_BAO','plikHM_TT_lowTEB_lensing_post_zre6p5','plikHM_TT_lowTEB_lensing_post_BAO_H070p6_JLA','plikHM_TT_tau07plikHM_TT_lowTEB_lensing_BAO','plikHM_TT_lowTEB_lensing_BAO_post_H070p6','plikHM_TT_lowTEB_lensing_BAO_post_H070p6_JLA','plikHM_TTTEEE_lowTEB','plikHM_TTTEEE_lowTEB_post_BAO','plikHM_TTTEEE_lowTEB_post_lensing','plikHM_TTTEEE_lowTEB_post_H070p6','plikHM_TTTEEE_lowTEB_post_JLA','plikHM_TTTEEE_lowTEB_post_zre6p5','plikHM_TTTEEE_lowTEB_post_BAO_H070p6_JLA','plikHM_TTTEEE_lowTEB_post_lensing_BAO_H070p6_JLA','plikHM_TTTEEE_lowl_lensing','plikHM_TTTEEE_lowl_lensing_post_BAO','plikHM_TTTEEE_lowl_lensing_post_BAO_H070p6_JLA','plikHM_TTTEEE_lowTEB_lensing']

Ndatasets=len(DataSets)

ImSamples = ['','BAO','BAO_H070p6_JLA','BAO_zre6p5','H070p6','JLA','reion','zre6p5']


Models={}
Models['model']=['base','base_omegak','base_Alens','base_Alensf','base_nnu','base_mnu','base_nrun','base_r','base_w','base_alpha1','base_Aphiphi','base_yhe','base_mnu_Alens','base_mnu_omegak','base_mnu_w','base_nnu_mnu','base_nnu_r','base_nnu_yhe',
'base_w_wa','base_nnu_meffsterile','base_nnu_meffsterile_r']
Models['npars']=6+np.array([len(x.split('_')) for x in Models['model']])


#------------------
model_list=Models['model']
data_list=DataSets
ims_list=ImSamples
print('------------')
#print('data:',data_list)
#print('model:',model_list)
#print('nparams:',Models['npars'])
print('Len data, model',len(data_list),len(model_list))
print('------------')

#-----------------------------
from mpi4py import MPI

def mpi_load_balance(MpiSize,nload):
    nmpi_pp=np.zeros(MpiSize,dtype=np.int)
    nmpi_pp[:]=nload/MpiSize
    r=nload%MpiSize
    if r != 0:
        nmpi_pp[1:r-1]=nmpi_pp[1:r-1]+1

    return nmpi_pp

#
main_loop_list=data_list
#
mpi_size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
comm = MPI.COMM_WORLD
#
nload=len(main_loop_list)
lpp=mpi_load_balance(mpi_size,nload)
#
#--------- outputs ----------

fout='{}/pla_evidence_table.txt'.format(outdir)
#amode=MPI.MODE_WRONLY
#fhandle = MPI.File.Open(comm, fout, amode)
if rank==0:
    fhandle=open(fout, 'w')
    fhandle.write('\n')
    fhandle.write('############## RootDirectory={} ########\n'.format(rootdir))
    fhandle.write('\n')

#------------------

all_mm={}
mce_cols=['k%s'%k for k in range(1,kmax)]
mce_info_cols=['NparamsCosmo','NparamsMC','Nsamples','MaxAutoCorrLen']


for ipp in range(lpp[rank]): 
    ig=ipp+lpp[0:rank-1].sum()
    kk=main_loop_list[ig]
    print('*** mpi_rank, idd, loop_key',rank, ig, kk)

    idd=ig
    dd=kk
    dd_dir=dd.split('_post_')[0]
    dd_name=dd #dd.split('plikHM_')[0]    
        
    mce=np.zeros((len(model_list),len(mce_cols)))
    mce_info=np.zeros((len(model_list), len(mce_info_cols) ),dtype=np.int) 
    
    for imm,mm in enumerate(model_list): #model loop            
        mm_name=mm.split('_')[0]        
        #
        method='{0}/{1}/{2}/{3}'.format(rootdir,mm,dd_dir,'%s_%s'%(mm,dd))
        if os.path.exists(method+'_1.txt'):            
            e,info = MCEvidence(method,ndim=Models['npars'][imm],
                                kmax=kmax,verbose=verbose,
                                thin=False).evidence(info=True)
            mce[imm,0:len(e)]=e
            mce_info[imm,0]=int(info['NparamsCosmo'])
            mce_info[imm,1]=int(info['NparamsMC'])
            mce_info[imm,2]=int(info['Nsamples_read'])
            mce_info[imm,3]=int(info['MaxAutoCorrLen'])
        else:
            print('*** not available: ',method)
            mce[imm,:]=np.nan
            mce_info[imm,:]=0
    #
    if not np.all(np.isnan(mce)):
        df = pd.DataFrame(mce,index=model_list,columns=mce_cols)
        df['delta_lnE_k1'] =mce[:,0] - np.nanmax(mce[:,0])
        for ik,k in enumerate(mce_info_cols):
            df[k]=mce_info[:,ik]
            
        print('--------------- data={}---------'.format(dd))
        print(tabulate(df, headers='keys', tablefmt='psql',floatfmt=".1f", numalign="left"))
        
        #append all tables to file
        df_all = comm.gather(df, root=0)
        dd_all = comm.gather(dd, root=0)        
        if rank==0:
            for ddi, dfi in zip(dd_all, df_all):
                fhandle.write('\n')                
                fhandle.write('************ data={} ************'.format(ddi))
                fhandle.write('\n')                
                fhandle.write(tabulate(dfi, headers='keys', tablefmt='psql'))
                fhandle.write('\n')
    
    #
    all_mm[dd]=df

#---------------------------------------------
#--------- gather all -----------------------
all_mm = comm.gather(all_mm, root=0)
if rank==0:
    #print('after gather type(all_mm)=',type(all_mm))
    all_mm={ k: v for d in all_mm for k, v in d.items() }
    print ('after_gather and concat: all_mm.keys:',all_mm.keys())

    # big_df=pd.DataFrame()
    # for mm,df in all_mm.items():
    #     big_df[mm]=df.loc['k1']
    # big_df=big_df.T
    #

    # Save a dictionary into a pickle file.
    fout_pkl='{0}/pla_evidence_df_dataKey_dict.pkl'.format(outdir)
    pickle.dump(all_mm, open(fout_pkl, "wb") )
    
    ##read
    #all_mm=pickle.load( open(fout_pkl, "rb" ) )

#---------------------------------------------

#big_df.to_latex('{0}/pla_evidence_df.tex'.format(outdir))

# Nmodels   = 21
# Ndatasets = 25

# Nparams   = np.zeros(Nmodels)
# Models    = np.empty( (Nmodels), dtype=[('model',object),('npars',int)] )
# DataSets  = np.empty( (Ndatasets), dtype=[('data',object)] )

# DataSets_v1=['planck_lowl','planck_lowl_lowLike',
#            'planck_lowl_lowLike_highL' ,'planck_tauprior',
#            'planck_tauprior_highL' ,'WMAP']

