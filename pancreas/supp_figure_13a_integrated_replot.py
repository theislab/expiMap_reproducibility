# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python rpy2_3
#     language: python
#     name: rpy2_3
# ---

# %% [markdown]
# # Replot
# Regenerate UMAP plots from integrated data.

# %%
import scanpy as sc
import numpy as np
from matplotlib import rcParams
from pathlib import Path
import pandas as pd

# %%
path_data='/lustre/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/qtr/integrated/gsCellType_query/querySTZ_rmNodElimination/mo/'
path_save=path_data+'figures/'
path_save_data=path_save+'data/'
params='_a_p0.0r1.0-akl_0.1-aklq_0.1-ra_0-uh_0-b_study_sample-sd_False-dp_0.01-lr_0.001-hls_830.830.830-es_1-nh_10000-ne_500-dll_softplus-ule_False-mg_200-mig_3-aea_100-aeaq_None-wd_0.0'

# %%
adata=sc.read(path_data+'adata_integration_RefQueryTraining'+params+'.h5ad')

# %% [markdown]
# Randomise cell order for plotting

# %%
# Randomise cell order
np.random.seed(0)
random_indices=np.random.permutation(list(range(adata.shape[0])))
adata=adata[random_indices,:]

# %% [markdown]
# Pretify some names and colours for plotting.

# %%
adata.obs.rename({'study':'Study','ref_query':'Reference and query',
                  'cell_type':'Cell type'},axis=1,inplace=True)

# %%
adata.obs['Reference and query']=adata.obs['Reference and query'].map(
    {'ref':'reference','query':'query'})

# %%
adata.obs['Study']=[f'%s (%s)'%(study, ref_query) 
                    for study,ref_query in 
                    zip(adata.obs['Study'],adata.obs['Reference and query'])]

# %%
adata.uns['Reference and query_colors']=['yellowgreen','darkmagenta']
adata.uns['Study_colors']=['#D81B60','#1E88E5','#FFC107','#004D40']

# %%
COLOR=['Study','Reference and query', 'Cell type']
def plot_integrated(adata,name,color=COLOR):
    sc._settings.ScanpyConfig.figdir=Path(path_save)
    rcParams['figure.figsize']= (8,8)
    sc.pl.umap(adata,color=color,
               wspace=0.6,
               #ncols=1,hspace=0.8,
               size=10,save='latent_'+name+'.png',show=False ,frameon=False,sort_order=False )


# %% [markdown]
# ## Ref+Query model with Ref+Query training cells

# %%
for emb in ['X_qtr', 'X_scvi','X_seurat','X_symphony']:
    # Compute neighbours and UMAP
    sc.pp.neighbors(adata, use_rep=emb)
    sc.tl.umap(adata)
    # Plot integrated embedding
    plot_integrated(adata,name='refqueryTraining_'+emb)
    pd.concat([
        pd.DataFrame(adata.obsm['X_umap'],index=adata.obs_names,columns=['UMAP1','UMAP2']),
        adata.obs[COLOR]],
        axis=1).to_csv(path_save_data+'latent_refqueryTraining_'+emb+'.tsv',sep='\t')

# %% [markdown]
# Make single xlsx from the embeddings of individual methods

# %%
writer = pd.ExcelWriter(path_save_data+'latent_refqueryTraining.xlsx',engine='xlsxwriter') 
for emb in ['X_qtr', 'X_scvi','X_seurat','X_symphony']:
    pd.read_table(path_save_data+'latent_refqueryTraining_'+emb+'.tsv',index_col=0,
                     ).to_excel(writer, sheet_name=emb)   
writer.save()

# %%
