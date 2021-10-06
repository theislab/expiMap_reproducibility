# +
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from matplotlib import rcParams
import pickle
import torch
import sys

import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
# -

path_data='/storage/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/qtr/'
# This was set to ending in querySpikeinDrug/ when we used all studies and different query
path_save_base=path_data+'integrated/gsCellType_query/querySTZ_rmNodElimination/'
path_gmt='/storage/groups/ml01/code/karin.hrovatin//qtr_intercode_reproducibility-/metadata/'

# +
parser = argparse.ArgumentParser()
def intstr_to_bool(x):
    return bool(int(x))
def hidden_layers_parser(hls):
    return [int(i) for i in hls.split(',')]
def parse_alpha(alpha):
    if ',' in alpha:
        alpha=[float(a) for a in alpha.split(',')]
    else:
        alpha=float(alpha)
    return alpha
        
parser.add_argument('-d','--dir',type=str,
                   help='Subdir added to base path for saving')
parser.add_argument('-a','--alpha', type=parse_alpha,default=None,
                    help='Training alpha') 
parser.add_argument('-akl','--alpha_kl',  type=float,
                    help='Training alpka_kl')
parser.add_argument('-aklq','--alpha_kl_query',  type=float,default=None,
                    help='Training alpka_kl for query integration. By default equal to alpha_kl.')
parser.add_argument('-ra','--remove_ambient',type=intstr_to_bool,
                   help='Remove ambient or not')
parser.add_argument('-uh','--use_hvg',type=intstr_to_bool,
                   help='Use HVG or not.')
parser.add_argument('-nh','--n_hvg',default=10000,type=int,
                   help='N HVG if used.')
parser.add_argument('-b','--batch',type=str,
                   help='Batch obs col name')
parser.add_argument('-sd','--subset_data',default=False,type=intstr_to_bool,
                    help='Use only data subset')
parser.add_argument('-dp','--data_proportion', default=0.01, type=float,
                    help='Proportion of data to use if subset_data=True')
parser.add_argument('-lr','--learning_rate', default=1e-3, type=float,
                    help='Training learning rate')
parser.add_argument('-hls','--hidden_layer_sizes', default=[2048,2048,2048], 
                    type=hidden_layers_parser,
                    help='Hidden layer sizes as comma-spearated string')
parser.add_argument('-es','--early_stopping', default=True, type=intstr_to_bool,
                    help='Use early stopping')
parser.add_argument('-ne','--n_epochs', default=400, type=int,
                    help='Max N epochs for training')
parser.add_argument('-s','--save', default=False, type=intstr_to_bool,
                    help='Save model and latent data')
parser.add_argument('-dll','--decoder_last_layer',type=str,default='softmax',
                   help='Last layer type of decoder')
parser.add_argument('-ule','--use_l_encoder',type=intstr_to_bool,default=False,
                   help='Use library size encoder')
parser.add_argument('-mg','--max_genes',type=int,default=None,
                   help='Largest gene set size to use. Default is None (no limit).'+\
                    'Should be int otherwise')
parser.add_argument('-mig','--min_genes',type=int,default=5,
                   help='Smallest gene set size to use.' )
parser.add_argument('-aea','--alpha_epoch_anneal',type=int,default=None,
                   help='N epochs for alpha annealing. Default is None.'+\
                    'Should be int otherwise')
parser.add_argument('-aeaq','--alpha_epoch_anneal_query',type=int,default=None,
                   help='N epochs for alpha annealing in query. Default is None.'+\
                    'Should be int otherwise')
parser.add_argument('-wd','--weight_decay',type=float,default=0.04,
                   help='Weight decay')


# +
# Just for testing without command line or saving data

# testing
if False:
    args= parser.parse_args(args=[
        '-d','test/',
        '-a','0.1,0.9',
        '-akl','10',
        '-ra','0',
        '-uh','0',
        '-b','study_sample',
        '-sd','1',
        '-nh','5000',
        '-ne','10',
        '-s','1',
        '-ule','1',
        '-mg','200',
    ])
# Read command line args
else:
    print(sys.argv)
    args = parser.parse_args()

# -

path_save=path_save_base+args.dir

args_name='a_'+str(args.alpha).replace(' ','').replace('[','p').replace(']','').replace(',','r')+\
    '-akl_'+str(args.alpha_kl)+\
    '-aklq_'+str(args.alpha_kl_query)+\
    '-ra_'+str(int(args.remove_ambient))+\
    '-uh_'+str(int(args.use_hvg))+'-b_'+str(args.batch)+\
    '-sd_'+str(args.subset_data)+'-dp_'+str(args.data_proportion)+\
    '-lr_'+str(args.learning_rate)+\
    '-hls_'+'.'.join([str(s) for s in args.hidden_layer_sizes])+\
    '-es_'+str(int(args.early_stopping))+\
    '-nh_'+str(args.n_hvg)+\
    '-ne_'+str(args.n_epochs)+\
    '-dll_'+str(args.decoder_last_layer)+\
    '-ule_'+str(args.use_l_encoder)+\
    '-mg_'+str(args.max_genes)+\
    '-mig_'+str(args.min_genes)+\
    '-aea_'+str(args.alpha_epoch_anneal)+\
    '-aeaq_'+str(args.alpha_epoch_anneal_query)+\
    '-wd_'+str(args.weight_decay)
print('Args:',args_name)

# ## Prepare ref and query

# Load data
adata=sc.read(path_data+'adata_annotated.h5ad')
print('Whole shape (non-raw):',adata.shape)

# Omit study from ref and query
study_omit='NOD_elimination'# remove this study 
if study_omit is not None:
    adata=adata[adata.obs.study!=study_omit,:]
    print('Whole filtered shape (non-raw):',adata.shape)

# Query info
ct_query='immune' # Cell types containing this word are removed from reference (but not query)
study_query='STZ' # remove this study from reference and use as query
adata.obs['ref_query']=['ref' if study!=study_query else 'query'
                           for study in adata.obs.study]

# +
# Ref
# Subset to all but query study
adata_r=adata.raw.to_adata()[adata.obs.study!=study_query,:]
adata_r.obs['cell_type']=adata[adata_r.obs_names,:].obs.cell_type
print('Ref shape:',adata_r.shape)
# Query-specific cells and their dublets not included in ref
adata_r=adata_r[~adata_r.obs.cell_type.str.contains(ct_query),:]
# Remove genes expressed in < 20 cells in ref
adata_r=adata_r[:,(adata_r.X!=0).sum(axis=0)>=20].copy()

# Remove top ambient genes
if args.remove_ambient:
    ambient=pd.read_table(path_data+'ambient_genes_scores.tsv',index_col=0)
    ambient=set(ambient[(ambient>0.005).any(axis=1)].index)
    adata_r=adata_r[:,[g for g in adata_r.var_names if g not in ambient]]

# Compute HVG across batches
if args.use_hvg:
    # Normalise data for HVG computation
    adata_r_norm=adata_r.copy()
    sc.pp.normalize_total(adata_r_norm, target_sum=1e6, exclude_highly_expressed=True)
    sc.pp.log1p(adata_r_norm)
    # HVG compute and subset
    adata_r=adata_r[:,sc.pp.highly_variable_genes(
        adata_r_norm, flavor='cell_ranger',n_top_genes=args.n_hvg,
        batch_key=args.batch,subset=False, inplace=False)['highly_variable']].copy()
    del adata_r_norm

print('Ref filtered shape:',adata_r.shape)

# +
# Add gene set anno to ref
# Omit gene sets not having at least 5 genes or too many genes
sca.add_annotations(adata_r, 
                    [path_gmt+'PanglaoDB_markers_27_Mar_2020_mouseEID.gmt',
                    path_gmt+'c2.cp.reactome.v4.0_mouseEID.gmt'], 
                    min_genes=args.min_genes, max_genes=args.max_genes, clean=False)
print('N used gene sets:',adata_r.varm['I'].shape[1])

# Subset data to only genes in added gene sets
adata_r=adata_r[:,adata_r.varm['I'].sum(axis=1)>0].copy()
print('N retained genes',adata_r.shape[1])
# -

# Query data - subset to query study cells and ref genes
adata_q=adata.raw.to_adata()[adata.obs.study==study_query,adata_r.var_names]
adata_q.obs['cell_type']=adata[adata_q.obs_names,:].obs.cell_type
adata_q.uns['terms']=adata_r.uns['terms'].copy()
print('Query shape:',adata_q.shape)


# Prepare data objects used in training

# Use whole datasets or only subset for testing out the scripts
def subset_data(adata,proportion=args.data_proportion):
    np.random.seed(0)
    random_indices=np.random.permutation(list(range(adata.shape[0]))
                                        )[:int(adata.shape[0]*proportion)]
    return adata[random_indices,:].copy()
if args.subset_data:
    adata_r_sub=subset_data(adata_r)
    adata_q_sub=subset_data(adata_q)
    adata_sub=subset_data(adata)
else:
    adata_r_sub=adata_r.copy()
    adata_q_sub=adata_q.copy()
    adata_sub=adata.copy()
adata_sub=adata_sub.raw.to_adata()[:,adata_r_sub.var_names]
print('Ref:',adata_r_sub.shape,'Query:',adata_q_sub.shape,'All:',adata_sub.shape)

# Save data for other integration methods
# Subset only to query and ref, not all cells
if args.save:
    adata_integration=adata_sub[adata_r_sub.obs_names.to_list()+\
                                adata_q_sub.obs_names.to_list(),:].copy()
    # Query-ref info
    adata_integration.obs.loc[adata_r_sub.obs_names,'ref_query']='ref'
    adata_integration.obs.loc[adata_q_sub.obs_names,'ref_query']='query'
    # Save
    print('Integration data shape:',adata_integration.shape)
    adata_integration.write(path_save+'adata_integration_RefQueryTraining_'+args_name+'.h5ad')
    print(path_save+'adata_integration_RefQueryTrainingg_'+args_name+'.h5ad')
    del adata_integration

    # Save terms info
    pickle.dump(np.array(adata_r.uns['terms']),open(
        path_save+'terms_'+args_name+'.pkl',
        'wb'))

# ## Training

# Set alpah and omegas
if isinstance(args.alpha,float) or args.alpha is None:
    alpha=args.alpha
    omega=None
else:
    omega = torch.full((len(adata_r.uns['terms']),1),fill_value=float(1)).view(-1)
    is_panglao=[t.startswith('PANGLAO') for t in adata_r.uns['terms']]
    is_reactome=[t.startswith('REACTOME') for t in adata_r.uns['terms']]
    omega[is_panglao] = args.alpha[0]
    omega[is_reactome] = args.alpha[1]
    alpha=float(1)

# reset query alpha_kl if unspecified
if args.alpha_kl_query is None:
    args.alpha_kl_query=args.alpha_kl

# ### Create TRVAE model and train it on reference dataset

model = sca.models.TRVAE(
    adata=remove_sparsity(adata_r_sub),
    condition_key=args.batch,
    hidden_layer_sizes=args.hidden_layer_sizes,
    use_mmd=False,
    recon_loss='nb',
    #beta=0.1,
    mask=adata_r.varm['I'].T,
    use_decoder_relu=False,
    mmd_instead_kl=False,
    decoder_last_layer=args.decoder_last_layer,
    use_l_encoder=args.use_l_encoder,
)

early_stopping_kwargs = {
    "early_stopping_metric": "val_loss",
    "threshold": 0,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}
model.train(
    lr=args.learning_rate,
    n_epochs=args.n_epochs,
    alpha_epoch_anneal=args.alpha_epoch_anneal, 
    alpha=alpha, # Higehr more integration
    omega=omega,
    alpha_kl=args.alpha_kl, # Higehr more integration
    weight_decay=args.weight_decay, 
    early_stopping_kwargs=early_stopping_kwargs,
    use_early_stopping=args.early_stopping, 
    seed=0
)

# Mark inactivated terms
adata_r_sub.uns['terms_is_active'] = \
    (model.model.decoder.L0.expr_L.weight.data.norm(p=2, dim=0)>0).cpu().numpy()
print('Inactive terms in ref model:')
inactive_terms=[term for i, term in enumerate(adata_r_sub.uns['terms']) 
       if not adata_r_sub.uns['terms_is_active'][i]]
print('N inactive:',len(inactive_terms))
print(inactive_terms)

# Add integrated embedding
adata_r_sub.obsm['X_integrated'] = model.get_latent(
    remove_sparsity(adata_r_sub).X, 
    adata_r_sub.obs[args.batch],mean=True
   )[:, adata_r_sub.uns['terms_is_active']]

# Compute neighbours and UMAP
sc.pp.neighbors(adata_r_sub, use_rep='X_integrated')
sc.tl.umap(adata_r_sub)


def plot_integrated(adata_temp,name,color=['study', 'cell_type']):
    random_indices=np.random.permutation(list(range(adata_temp.shape[0])))
    sc._settings.ScanpyConfig.figdir=Path(path_save)
    rcParams['figure.figsize']= (8,8)
    sc.pl.umap(adata_temp[random_indices,:],color=color,
               wspace=0.6,
               #ncols=1,hspace=0.8,
               size=10,save='latent_'+name+'.png',show=False ,frameon=False )


# Plot integrated UMAP
plot_integrated(adata_r_sub,name='ref_'+args_name)

# Save model
if args.save:
    model.save(path_save+'ref_'+args_name+'/',overwrite=True)
    print('Saved ref model in:',path_save+'ref_'+args_name+'/')

# Save integrated embedding
if args.save:
    pd.DataFrame(adata_r_sub.obsm['X_integrated'],
                 index=adata_r_sub.obs_names,
                 columns=np.array(adata_r_sub.uns['terms'])[adata_r_sub.uns['terms_is_active']]
                ).to_csv(
        path_save+'latent_ref_'+args_name+'.tsv',sep='\t'
        )

# ### Add query

# Make query model from original model
model_q = sca.models.TRVAE.load_query_data(remove_sparsity(adata_q_sub), model)

# Train query model
model_q.train(
    lr=args.learning_rate,
    n_epochs=args.n_epochs,
    alpha_epoch_anneal=args.alpha_epoch_anneal_query,  
    alpha_kl=args.alpha_kl_query, 
    early_stopping_kwargs=early_stopping_kwargs,
    use_early_stopping=args.early_stopping, 
    weight_decay=args.weight_decay, 
    seed=0
)

# Mark inactivated terms
adata_q_sub.uns['terms_is_active'] = \
    (model_q.model.decoder.L0.expr_L.weight.data.norm(p=2, dim=0)>0).cpu().numpy()
print('Inactive terms in ref+query model:')
inactive_terms=[term for i, term in enumerate(adata_q_sub.uns['terms']) 
       if not adata_q_sub.uns['terms_is_active'][i]]
print('N inactive:',len(inactive_terms))
print('List of inactive terms:', inactive_terms)

# Save model
if args.save:
    model_q.save(path_save+'refquery_'+args_name+'/',overwrite=True)
    print('Saved refquery model in:',path_save+'refquery_'+args_name+'/')

# ### Prediction Q&R with Q model

# #### Training cells
# Prediction from query adapated model for training data: Q & R (excluding the previously ommited cell type).

# Subset to training data
adata_training=adata.raw.to_adata()[adata_r_sub.obs_names.to_list()+\
                            adata_q_sub.obs_names.to_list(),adata_sub.var_names]

# Prediction 
adata_training.obsm['X_integrated'] = model_q.get_latent(
    remove_sparsity(adata_training).X, 
    adata_training.obs[args.batch], mean=True
    )[:, adata_q_sub.uns['terms_is_active']]

# Compute neighbours and UMAP
sc.pp.neighbors(adata_training, use_rep='X_integrated')
sc.tl.umap(adata_training)

# Plot integrated embedding
plot_integrated(adata_training,name='refqueryTraining_'+args_name,
                color=['study','cell_type','ref_query'])

# Save integrated embedding
if args.save:
    pd.DataFrame(adata_training.obsm['X_integrated'],
                 index=adata_training.obs_names,
                 columns=np.array(adata_q_sub.uns['terms'])[adata_q_sub.uns['terms_is_active']]
                ).to_csv(
        path_save+'latent_refqueryTraining_'+args_name+'.tsv',sep='\t'
        )

# #### All cells
# Prediction from query adapated model for all data: Q & R (including the previously ommited cell type).

# Prediction 
adata_sub.obsm['X_integrated'] = model_q.get_latent(
    remove_sparsity(adata_sub).X, 
    adata_sub.obs[args.batch], mean=True
    )[:, adata_q_sub.uns['terms_is_active']]

# Compute neighbours and UMAP
sc.pp.neighbors(adata_sub, use_rep='X_integrated')
sc.tl.umap(adata_sub)

# Plot integrated embedding
plot_integrated(adata_sub,name='refquery_'+args_name,color=['study','cell_type','ref_query'])

# Save integrated embedding
if args.save:
    pd.DataFrame(adata_sub.obsm['X_integrated'],
                 index=adata_sub.obs_names,
                 columns=np.array(adata_q_sub.uns['terms'])[adata_q_sub.uns['terms_is_active']]
                ).to_csv(
        path_save+'latent_refquery_'+args_name+'.tsv',sep='\t'
        )

print('Finished!')
