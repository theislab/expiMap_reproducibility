# +
import pandas as pd
import argparse
import os
import numpy as np
import warnings 
import pickle
import time

from scIB import metrics as sm
import scanpy as sc

# +
parser = argparse.ArgumentParser()
parser.add_argument('-if','--input_file', 
                    help='Absolute path of analysed adata file.' + 
                    'Should have QC and normalised log transformed counts in X '+
                    '(e.g. what you would usually use for downstream analyses)'+
                    'and integrated embedding in obsm.') 
parser.add_argument('-ie','--integrated_embedding', 
                    help='Name of integrated embedding slot.')
parser.add_argument('-b','--batch',
                   help='Batch obs col name')
parser.add_argument('-ct','--cell_type',
                   help='Cell type obs col name')

def intstr_to_bool(x):
    return bool(int(x))
parser.add_argument('-pr','--PC_regression',type=intstr_to_bool, default=True)
parser.add_argument('-ab','--ASW_batch',type=intstr_to_bool, default=True)
parser.add_argument('-k','--kBET',type=intstr_to_bool, default=True)
parser.add_argument('-gc','--graph_connectivity',type=intstr_to_bool, default=True)
parser.add_argument('-gil','--graph_iLISI',type=intstr_to_bool, default=True)
parser.add_argument('-gcl','--graph_cLISI',type=intstr_to_bool, default=True)
parser.add_argument('-n','--NMI',type=intstr_to_bool, default=True)
parser.add_argument('-a','--ARI',type=intstr_to_bool, default=True)
parser.add_argument('-act','--ASW_cell_type',type=intstr_to_bool, default=True)
# -

args = parser.parse_args()

# For testing
if False:
    args= parser.parse_args(args=[
        '-if','/storage/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/qtr/data_integrated_try.h5ad',
        '-ie','X_integrated',
        '-b','study_sample',
        '-ct','cell_type'
    ])


# ## Prepare data

# Load adata
adata=sc.read(args.input_file)


# +
# prepare data for scIB
def remove_neigh(adata):
    # Remove any neighbour related entries from  adata
    adata.obsp.pop('connectivities',None)
    adata.obsp.pop('distances',None)
    adata.uns.pop('neighbors',None)
    
# Remove raw if present
if adata.raw is not None:
    del adata.raw

# Unintegrated adata
adata_full=adata.copy()
remove_neigh(adata_full) # Neighbours are computed within the one metric that uses this
# Integrated adata
latent_adata=adata.copy()
remove_neigh(latent_adata)
latent_adata.obsm['X_emb']=latent_adata.obsm[args.integrated_embedding].copy()
latent_adata = sc.pp.neighbors(latent_adata, n_pcs=0, use_rep='X_emb', copy=True)
# -

# Prepare opt clusters for NMI/ARI
res_max, nmi_max, nmi_all = sm.opt_louvain(adata=latent_adata,
        label_key=args.cell_type, cluster_key='opt_louvain', function=sm.nmi,
        plot=False, verbose=False, inplace=True, force=True)   

# ## Metrics

# Dict for saving metrics
metrics={}

# ### Batch

# Principal component regression
if args.PC_regression:
    print('Computing PC_regression')
    metrics['PC_regression']=sm.pcr_comparison(adata_pre=adata_full, 
                                               adata_post=latent_adata, 
                                               covariate=args.batch,
                                        embed='X_emb')

# Batch ASW
if args.ASW_batch:
    print('Computing ASW_batch')
    metrics['ASW_batch']=sm.silhouette_batch(latent_adata, 
                                             batch_key=args.batch,
                                             group_key=args.cell_type,
                                             metric='euclidean',  embed='X_emb', 
                                             verbose=False, return_all=False,
                         )

# kBET
if args.kBET:
    print('Computing kBET')
    try:
        metrics['kBET']=sm.kBET(adata=latent_adata, batch_key=args.batch, 
                               label_key=args.cell_type, 
                                         embed='X_emb', 
                                     type_ = 'embed',
                                scaled=True,
                               verbose=False) 

    except sm.NeighborsError as err:
        metrics['kBET']=0
        warnings.warn('kBET can not be calculated and was given value 0:')
        print("ValueError error: {0}".format(err))

# Graph connectivity
if args.graph_connectivity:
    print('Computing graph_connectivity')
    metrics['graph_connectivity']=sm.graph_connectivity(adata=latent_adata, 
                                                        label_key=args.cell_type)

# Graph  iLISI
if args.graph_iLISI:
    print('Computing graph_iLISI')
    try:
        metrics['graph_iLISI'] = sm.ilisi_graph(
            adata=latent_adata, batch_key=args.batch, 
             type_= 'embed',
                                  subsample=0.5*100, scale=True,
                                        multiprocessing = True,verbose=False)

    except FileNotFoundError as err:
        warnings.warn('Could not compute iLISI scores due to FileNotFoundError')
        metrics['graph_iLISI'] =np.nan
        print("FileNotFoundError: {0}".format(err))

# ### BIO

# Graph  cLISI
if args.graph_cLISI:
    print('Computing graph_cLISI')
    try:
        metrics['graph_cLISI'] = sm.clisi_graph(
            adata=latent_adata, batch_key=args.batch, 
            label_key=args.cell_type,  type_= 'embed',
                                  subsample=0.5*100, scale=True,
                                        multiprocessing = True, verbose=False)

    except FileNotFoundError as err:
        warnings.warn('Could not compute cLISI scores due to FileNotFoundError')
        metrics['graph_cLISI'] =np.nan
        print("FileNotFoundError: {0}".format(err))

# NMI
if args.NMI:
    print('Computing NMI')
    metrics['NMI']=sm.nmi(adata=latent_adata, 
                          group1=args.cell_type, group2='opt_louvain', 
                            method="arithmetic",nmi_dir=None)

# ARI
if args.ARI:
    print('Computing ARI')
    metrics['ARI']=sm.ari(adata=latent_adata, 
                          group1=args.cell_type, group2='opt_louvain')

# ASW cell type
if args.ASW_cell_type:
    print('Computing ASW_cell_type')
    metrics['ASW_cell_type']=sm.silhouette(adata=latent_adata, 
                                           group_key=args.cell_type, embed='X_emb', 
                                           metric='euclidean')

# ### Save metrics

metrics['adata_name']=args.input_file.split('/')[-1]
path_save='/'.join(args.input_file.split('/')[:-1])+\
                '/scIB_IE_'+args.integrated_embedding+\
                '_B_'+args.batch+'_CT_'+args.cell_type+'_uid'+str(time.time())+'.pkl'
pickle.dump(metrics,open(path_save,'wb'))
print('Saved to:',path_save)
