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
#     display_name: qtr
#     language: python
#     name: qtr
# ---

# %%
import scanpy as sc
import pandas as pd
import pickle

import scarches as sca # import also numpy and torch
import torch  # uses already imported torch, can set random seed latter
import numpy as np # uses already imported numpy, can set random seed latter

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch

from sklearn.metrics import f1_score

from scarches.dataset.trvae.data_handling import remove_sparsity

import sys
sys.path.insert(0, '/storage/groups/ml01/code/karin.hrovatin/qtr_intercode_reproducibility-/')
import scripts.annotation_transfer_utils as atu
import importlib
importlib.reload(atu)
import scripts.annotation_transfer_utils as atu

# %%
path_gmt='/storage/groups/ml01/code/karin.hrovatin//qtr_intercode_reproducibility-/metadata/'
path_data='/storage/groups/ml01/workspace/karin.hrovatin//data/pancreas/scRNA/qtr/integrated/gsCellType_query/querySTZ_rmNodElimination/'
subdir='mo/'
params_name='a_p0.0r1.0-akl_0.1-aklq_0.1-ra_0-uh_0-b_study_sample-sd_False-dp_0.01-lr_0.001-hls_830.830.830-es_1-nh_10000-ne_500-dll_softplus-ule_False-mg_200-mig_3-aea_100-aeaq_None-wd_0.0'
path_fig=path_data+subdir+'figures/'
path_res=path_data+subdir+'results/'

# %%
sc._settings.ScanpyConfig.figdir=Path(path_fig)
sc._settings.settings._vector_friendly=True

# %%
# Load adata used for integration
adata=sc.read(path_data+subdir+'adata_integration_RefQueryTraining_'+params_name+'.h5ad')
adata=remove_sparsity(adata)
adata.uns['terms']=pickle.load(open(path_data+subdir+'terms_'+params_name+'.pkl','rb'))

# %%
# Load model
model = sca.models.TRVAE.load(path_data+subdir+'refquery_'+params_name+'/', 
                              adata,map_location=torch.device('cpu'))

# %%
# Directions of terms in the model
directions = model._latent_directions(method="sum")

# %% [markdown]
# ## Cell types

# %% [markdown]
# ### Cell type-term enrichment per cell type
# Are cell types enriched for matched cell type terms? 
#
# Perform OvR enrichment of each cell type vs rest (OvR). To ensure that the analysed cell type is only in the O group we remove doublets and proliferative cell types taht would otherwise affect enrichment as part of R group when comparing to associated non-doublte or prloferative cell types.

# %%
# Select cell types for analysis and create cell name groups
# Use cell types that contain one of the below words in name as that cell type
cell_types=['acinar','alpha','beta','delta','ductal',
            'endothelial','gamma','immune','schwann','stellate']
# Exclude doublets and proliferative populations
ct_exclude=[ct for ct in adata.obs.cell_type.unique() 
            if 'proliferative' in ct or
           sum([c in ct for c in cell_types])>1]
print('Excluded cell types:',ct_exclude)

# Prepare cell groups for enrichment
ct_dict={}
for ct in cell_types:
    ct_dict[ct]=adata.obs_names[adata.obs.cell_type.str.contains(ct).values & 
                                ~adata.obs.cell_type.isin(ct_exclude).values].tolist()
# Use only Panglao terms
db='PANGLAO'
terms_idx=np.argwhere([t.startswith(db) for t in adata.uns['terms']]).ravel()
# Cell type scores
np.random.seed(0)
torch.manual_seed(0)
scores = model.latent_enrich(ct_dict, comparison="rest", directions=directions, adata=adata,
                            select_terms=terms_idx,n_perm=10000)

# Print out results - top scores (directed)
for ct in scores:
    # Count N cells in cell type
    print('\n',ct,'N cells:',len(ct_dict[ct]))
    data=pd.DataFrame({'bf':scores[ct]['bf']},
                      index=[t.replace(db+'_','') for t in adata.uns['terms'][terms_idx]])
    data.index.name='term'
    data=data.sort_values('bf',ascending=False)
    display(data.iloc[:10])

# %% [markdown]
# For most cell types the most highly enriched terms represent true cell types. In other cases these cell types can be infered based on knowldge about the organ. For example, acinar, gamma, and stellate enrichment tables contain the correct cell type further down the enrichment list and the cell types with higher ranking, such as hepatoblasts (hepatic progenitor cells), principal cells (located in kidney), and adipocytes, can be excluded due to lack of presence in pancreas. Namely, some cell types are similar across organs, leading to high scores for related cell type from different organs. This can be clearly seen for ductal cells, which are among others strongly enriched for cholangiocytes (bile duct epithelium), hepatoblasts (progenitors of hepatocyte and cholangiocytes), microfold cells (part of gut epithelium), and other epithelial cells. This information leads us to conclude that this cell cluster likely contains epithelial cells, such as ductal cells. To avoid interference of cell types from other tissues one could exclude non-organ related cell type terms before integration or enrichment.

# %% [markdown]
# ### Distribution of cell-type corresponding term scores across cell types
# Compare term-scores for cell types that could be present in pancreas tissue to manual annotation.

# %%
# Add latent embedding with directions, keep only active terms
adata.obsm['X_qtr_directed']=(model.get_latent(adata.X,  adata.obs['study_sample'], 
                                     mean=True ) * directions.reshape(1,-1))

# %%
# Terms to plot and ct association
terms_ct={'PANGLAO_ACINAR_CELLS':['acinar'],
          'PANGLAO_ALPHA_CELLS':['alpha'],
          'PANGLAO_BETA_CELLS':['beta'],
            'PANGLAO_DELTA_CELLS':['delta'],
          'PANGLAO_DUCTAL_CELLS':['ductal'],
          'PANGLAO_ENDOTHELIAL_CELLS':['endothelial'],
            'PANGLAO_GAMMA_(PP)_CELLS':['gamma'],
          'PANGLAO_PERI-ISLET_SCHWANN_CELLS':['schwann'],
            'PANGLAO_PANCREATIC_STELLATE_CELLS':['stellate_activated', 'stellate_quiescent'],
             # Different immune types
         'PANGLAO_T_MEMORY_CELLS':['immune'],
         'PANGLAO_B_CELLS_MEMORY':['immune'],
             'PANGLAO_DENDRITIC_CELLS':['immune'],
             'PANGLAO_MACROPHAGES':['immune'],
          'PANGLAO_EOSINOPHILS':['immune'],
          'PANGLAO_T_CELLS':['immune'],
          'PANGLAO_B_CELLS':['immune'],
          "PANGLAO_NEUTROPHILS":['immune'],
         'PANGLAO_MONOCYTES':['immune'],
         'PANGLAO_T_HELPER_CELLS':['immune'],
         'PANGLAO_NK_CELLS':['immune'],}


# %% [markdown]
# Distribution of cell type scores across manually annotated cell types.

# %%
# Plot term distn
rcParams['figure.figsize']=(10,4)
fig,axs=plt.subplots(len(terms_ct),1,figsize=(10,6*len(terms_ct)))
for idx,(term, cts) in enumerate(terms_ct.items()):
    cts=cts.copy() # Copy cts so that list is not modified in place
    ax=axs[idx]
    term_idx=np.argwhere(adata.uns['terms']==term)[0][0]
    adata.obs[term]=adata.obsm['X_qtr_directed'][:,term_idx]                                     
    sc.pl.violin(adata, keys=term, groupby='cell_type',stripplot=False,rotation=90,show=False,
                  ax=ax)
    # Mark ct median
    ct_median=adata.obs.query('cell_type in @cts')[term].median()    
    ax.axhline(ct_median,c='k')
    # Mark ct
    ct_idxs=[np.argwhere(adata.obs.cell_type.cat.categories.values==cti)[0,0] for cti in cts]
    for ct_idx in ct_idxs:
        ax.axvline(ct_idx,c='r')
    # Mark related cell types
    # Fix for stellate cells as doublets do not have subtype info
    if term=='PANGLAO_PANCREATIC_STELLATE_CELLS':
        cts.append('stellate')
    related_cts=[ct for ct in adata.obs.cell_type.unique()
                if any([c in ct for c in cts]) and ct not in cts]
    ct_idxs=[np.argwhere(adata.obs.cell_type.cat.categories.values==cti)[0,0] 
             for cti in related_cts]
    for ct_idx in ct_idxs:
        ax.axvline(ct_idx,c='orange')
    adata.obs.drop(term,axis=1,inplace=True)
fig.tight_layout()
plt.savefig(path_fig+'cellTypeScore_distributions.pdf',dpi=300)

# %% [markdown]
# Note: Endocrine proliferative cells have high scores for different endocrine cell types as they contain multiple endocrine cell types (including alpha, beta, delta). They are not marked as belonging to either of these cell types due to not being resolved on cell-type level.

# %%
# Umap on latent embedding; could filter out inactive terms
sc.pp.neighbors(adata, use_rep='X_qtr_directed')
sc.tl.umap(adata)

# %% [markdown]
# Comparison of manually annotated cell types and cell type scores on UMAP.

# %%
# Annotated cell types
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata,color='cell_type',s=10)

# %%
# Plot cell type term scores on UMAP
plot_terms=list(terms_ct.keys())
# Get term indices and sort terms in the same order
plot_terms_idx=[np.argwhere(adata.uns['terms']==t)[0,0] for t in plot_terms]
adata.obs[plot_terms]=adata.obsm['X_qtr_directed'][:,plot_terms_idx]
rcParams['figure.figsize']=(6,6)
# Set colormap values for plotting, no min/max
sc.pl.umap(adata,color=plot_terms,s=10,
           cmap='coolwarm',vcenter=0,
          save='_pancreasCT_scores.pdf'
          )
adata.obs.drop(plot_terms,axis=1,inplace=True)

# %% [markdown]
# For most of the cell types the regions of high cell type specific gene set scores correspond to prior cell type annotation. This also includes doublets, such as beta-delta and endothelial-stellate droplets. 
#
# Additionally, PanglaoDB gene set scores help us separate between different immune lineages and cell types that we did not resolve before.

# %% [markdown]
# ### Annotation transfer
# We can use the latent embedding for annotation transfer. Cell type term scores can be latter used to resolve clusters with high annotation-transfer uncertainity. 

# %%
# Prepare data for classification
# Remove inactive terms and non-panglao terms to speed up classification
term_is_active=(adata.obsm['X_qtr_directed']!=0).any(axis=0)
term_is_panglao=np.array([t.startswith('PANGLAO') for t in adata.uns['terms']])
terms_keep=term_is_active&term_is_panglao
# Get latent embedding of ref and query as adata
adata_latent_r = sc.AnnData(
    adata[adata.obs.ref_query=='ref',:].obsm['X_qtr_directed'][:,terms_keep],
    obs=adata[adata.obs.ref_query=='ref',:].obs)
adata_latent_r.var_names=adata.uns['terms'][terms_keep]
adata_latent_q = sc.AnnData(
    adata[adata.obs.ref_query=='query',:].obsm['X_qtr_directed'][:,terms_keep],
    obs=adata[adata.obs.ref_query=='query',:].obs)
adata_latent_q.var_names=adata.uns['terms'][terms_keep]
# Cell type column
ct_col='cell_type'

# %%
# Classification
atu.weighted_knn(adata_latent_r,adata_latent_q,label_key=ct_col,
             n_neighbors=10,threshold=0.8)

# %%
print('Weighted F1 score for query predictions:',round(f1_score(
        adata_latent_q.obs[ct_col], adata_latent_q.obs['pred_'+ct_col], average="weighted"),2))

# %%
# Compute embedding of query on terms used for classification
sc.pp.neighbors(adata_latent_q,use_rep='X')
sc.tl.umap(adata_latent_q)

# %%
# Comparison of predicted and manually annotated cell types
atu.plot_true_vs_pred(adata_latent_q=adata_latent_q,ct_col=ct_col)
plt.savefig(path_fig+'annotation_transfer_cellTypes.pdf',dpi=300)

# %%
# Prediction certainty
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata_latent_q,  color=["evaluation",'uncertainty'], frameon=False,  size=10,
          save='_annotation_transfer_uncertainty.pdf'
          )

# %% [markdown]
# Cell clusters with low prediction probability (e.g. immune) can be annotated based on cell type term scores, either with enrichment of cell clusters (as shown above) or with plots of score distributions of cell types expected to occur in the sample tissue.

# %%
# Query celltypes
# Prediction certainty
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata_latent_q,  color=["cell_type"], size=10,
          save='_query_celltypes.pdf'
          )

# %%
plot_terms=list(terms_ct.keys())
rcParams['figure.figsize']=(6,6)
# Set colormap values for plotting, no min/max
sc.pl.umap(adata_latent_q,color=plot_terms,s=10,cmap='coolwarm',vcenter=0,
           save='_query_pancreasCT_scores.pdf'
          )

# %% [markdown]
# This helps us to identify some cell types that were not predicted, such as immune cells (absent from reference) and acinar cells (low number of cells).

# %% [markdown]
# #### Cell type-term enrichment per cell cluster of query
# Try to annotate cell clusters based on cell type score enrichment.
# Perform OvR enrichment of each cell cluster vs rest (OvR) and report previously annotated main cell types and samples in that cluster.

# %%
# Cluster cells
sc.tl.leiden(adata_latent_q,resolution=2)

# %%
# Plot clusters
rcParams['figure.figsize']=(5,5)
sc.pl.umap(adata_latent_q,color=['cell_type','leiden'],wspace=0.9)

# %%
# Prepare cell groups for enrichment
ct_dict={}
for ct in adata_latent_q.obs.leiden.cat.categories:
    ct_dict[ct]=adata_latent_q.obs_names[adata_latent_q.obs.leiden == ct].tolist()
# Use only Panglao terms
db='PANGLAO'
terms_idx=np.argwhere([t.startswith(db) for t in adata.uns['terms']]).ravel()
# Cell type scores on query data
np.random.seed(0)
torch.manual_seed(0)
scores = model.latent_enrich(ct_dict, comparison="rest", directions=directions, 
                             adata=adata[adata.obs.ref_query=='query',:],
                            select_terms=terms_idx,n_perm=10000)

# Print out results - top scores (directed)
for ct in scores:
    # Count N cells in cell type
    print('\n Cluster',ct,'N cells:',len(ct_dict[ct]))
    # Main cell types in cluster
    ct_counts=adata_latent_q[ct_dict[ct],:].obs.cell_type.value_counts(normalize=True)
    display(ct_counts[ct_counts>0.2])
    # Main sample in cluster
    design_counts=adata_latent_q[ct_dict[ct],:].obs.design.value_counts(normalize=True)
    display(design_counts)
    
    # Scores display
    data=pd.DataFrame({'bf':scores[ct]['bf']},
                      index=[t.replace(db+'_','') for t in adata.uns['terms'][terms_idx]])
    data.index.name='term'
    display(data.sort_values('bf',ascending=False).iloc[:10])

# %% [markdown]
# Per cluster enrichments in many cases enable correct cluster annotation. However, there are some exceptions:
# - In some cases dublet status is not clear from the enrichment. However, when plotting cell scores on UMAP (as above) this can be easily resolved. 
# - Proliferative cells are not identified as we did not use a gene set specific for proliferative cells. Users could thus add proliferative markers gene set to analysis.
# - Beta cell clusters with cells comming from predominately healthy control sample had enrichment for beta cells. However, beta cell clusters containing mainly cells from STZ treated diseased samples were often not enriched for beta cell gene set. This is in accordance to known dedifferentiation and loss of identity of beta cells after STZ treatment. However, by looking at beta cell score on the above UMAPs one can clearly identify the beta cell cluster, showing variable beta cell score strength in accordance to STZ treatment (see also distribution of beta cell related scores vs sample status below). 
#
# The lack of apropriate enrichment in some clusters might be also due to the use of OvR enrichment, leading to presence of cells of the same cell type in both query and reference group at the same time. Thus we suggest users to rather plot cell type scores of interest on UMAP (see above) to help resolve cell type identity. 
#
# Furthermore, plotting cell type scores on UMAP can help to identify lack of clustering resolution, such as in our cluster 19. This cluster contain both alpha-immune doublets and immune cells, as can be clearly observed from the UMAPs showing cell type scores (above). Thus, score UMAPs can help us select appropriate clustering and sub-sclustering resolutions to ease the annotatiuon process.

# %% [markdown]
# ## Beta cell function - T2D model vs healthy
# Compare beta cells from healthy (control) and T2D-model (STZ-treated) samples from STZ study to find gene sets different in T2D.

# %%
# Make beta cell adata
adata_beta=adata[adata.obs.cell_type=='beta',:].copy()
sc.pp.neighbors(adata_beta, use_rep='X_qtr_directed')
sc.tl.umap(adata_beta)

# %% [markdown]
# ### Separation of healthy and STZ-treated beta cells

# %%
# Report cell as belonging to ref or one of the query designs
adata_beta.obs['ref_querySample']=pd.Categorical(pd.Series(
    [rq if rq=='ref' else design
    for rq, design in zip(adata_beta.obs.ref_query,adata_beta.obs.design)],
    dtype='category',index=adata_beta.obs_names),
     categories=['STZ','STZ_GLP-1','STZ_estrogen','STZ_GLP-1_estrogen',
                'STZ_insulin','STZ_GLP-1_estrogen+insulin','control','ref'],
     ordered=True)

# Make study+design column
adata_beta.obs['study_design']=[study+'_'+design for study,design in 
                                zip(adata_beta.obs['study'],adata_beta.obs['design'])]
adata_beta.obs['study_design']=adata_beta.obs['study_design'].astype('category')

# Set colors 
colormap={'STZ':'cyan',
          'STZ_GLP-1':'tab:purple','STZ_estrogen':'tab:pink','STZ_GLP-1_estrogen':'pink',
         'STZ_insulin':'orange','STZ_GLP-1_estrogen+insulin':'gold',
         'control':'yellowgreen', 'ref':'#bfbfbf'}
adata_beta.uns['ref_querySample_colors']=[
    colormap[cat] for cat in adata_beta.obs['ref_querySample'].cat.categories]
study_colors_map={'Fltp_P16':'#D81B60','NOD':'#1E88E5',
                  'STZ':'#FFC107','spikein_drug':'#004D40'}
adata_beta.uns['study_colors']=[study_colors_map[s] 
                                for s in adata_beta.obs['study'].cat.categories]

# Set same colors in latent query adata
adata_latent_q.obs['design']=pd.Categorical(
    adata_latent_q.obs.design,
     categories=[c for c in adata_beta.obs.ref_querySample.cat.categories 
                 if c in adata_latent_q.obs.design.unique()],
    ordered=True)
adata_latent_q.uns['design_colors']=[
    colormap[cat] for cat in adata_latent_q.obs['design'].cat.categories]

# %% [markdown]
# #### Loss of beta cell identity

# %% [markdown]
# Beta cells are known to be dedifferentiated and losse their identity in diabetes and upon STZ treatment. Thus we compare beta cell and and other cell type socres in healthy and STZ-treated beta cells. 
#
# Each plot highlights a subset of cells, the rest are marked in pale gray.

# %%
# Plot terms in beta cells
terms_plot=['BETA_CELLS','PANCREATIC_PROGENITOR_CELLS']
for term in terms_plot:
    adata_beta.obs[term]=adata_beta.obsm[
        'X_qtr_directed'][:, np.argwhere(adata_beta.uns['terms']=='PANGLAO_'+term)[0][0]]
fig,ax=plt.subplots(2,2,figsize=(12,8),sharey=True,sharex=True)
# Temporarily change p
sc.pl.scatter(adata_beta, x=terms_plot[0], y=terms_plot[1], 
              color='ref_querySample',groups='ref',ax=ax[0,0],show=False,
             palette=[c if c!='#bfbfbf' else 'k' 
                      for c in adata_beta.uns['ref_querySample_colors']]) 
ax[0,0].set_title('Reference')
sc.pl.scatter(adata_beta, x=terms_plot[0], y=terms_plot[1], 
              color='ref_querySample',
              groups=[group for group in adata_beta.obs.ref_querySample.cat.categories
                      if group!='ref'], ax=ax[0,1],show=False)
ax[0,1].set_title('Query')
sc.pl.scatter(adata_beta, x=terms_plot[0], y=terms_plot[1], 
              color='ref_querySample',groups='control',ax=ax[1,0],show=False) 
ax[1,0].set_title('Query healthy')
sc.pl.scatter(adata_beta, x=terms_plot[0], y=terms_plot[1], 
              color='ref_querySample',groups='STZ',ax=ax[1,1],show=False)
ax[1,1].set_title('Query STZ (diabetic)')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                    wspace=0.6, hspace=0.4)
adata_beta.obs.drop(terms_plot,axis=1,inplace=True)
fig.tight_layout()
plt.savefig(path_fig+'beta_identity_progenitor.pdf',dpi=300)

# %%
# Plot terms in beta cells
terms_plot=['BETA_CELLS','ENTEROENDOCRINE_CELLS']
for term in terms_plot:
    adata_beta.obs[term]=adata_beta.obsm[
        'X_qtr_directed'][:, np.argwhere(adata_beta.uns['terms']=='PANGLAO_'+term)[0][0]]
fig,ax=plt.subplots(2,2,figsize=(12,8),sharey=True,sharex=True)
sc.pl.scatter(adata_beta, x=terms_plot[0], y=terms_plot[1], 
              color='ref_querySample',groups='ref',ax=ax[0,0],show=False,
             palette=[c if c!='#bfbfbf' else 'k' 
                      for c in adata_beta.uns['ref_querySample_colors']]) 
ax[0,0].set_title('Reference')
sc.pl.scatter(adata_beta, x=terms_plot[0], y=terms_plot[1], 
              color='ref_querySample',
              groups=[group for group in adata_beta.obs.ref_querySample.cat.categories
                      if group!='ref'], ax=ax[0,1],show=False)
ax[0,1].set_title('Query')
sc.pl.scatter(adata_beta, x=terms_plot[0], y=terms_plot[1], 
              color='ref_querySample',groups='control',ax=ax[1,0],show=False) 
ax[1,0].set_title('Query healthy')
sc.pl.scatter(adata_beta, x=terms_plot[0], y=terms_plot[1], 
              color='ref_querySample',groups='STZ',ax=ax[1,1],show=False)
ax[1,1].set_title('Query STZ (diabetic)')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                    wspace=0.6, hspace=0.4)
adata_beta.obs.drop(terms_plot,axis=1,inplace=True)
fig.tight_layout()
plt.savefig(path_fig+'beta_identity_enteroendocrine.pdf',dpi=300)

# %% [markdown]
# We can see that STZ treated cells have lower beta cell score than healthy control. Similarly, they have higher scores for pancreatic progenitors and enteroendocrine cells. This agrees with previousn research on diabetes-related dedifferentiation of beta cells.
#
# We can also see that using multiple scores helps us to better distinguish between healthy and diseased populations than using a single score.

# %% [markdown]
# Below we check most enriched cell type terms separating healthy control and STZ treated query beta cells, which could indicate differentiation towards other cell types in diabetic condition.

# %%
# Compare WT and STZ beta cells from STZ study
condition_dict={
    'beta_STZ':adata.obs.query('study_sample=="STZ_G2" & cell_type=="beta"').index.tolist(),
    'beta_control':adata.obs.query('study_sample=="STZ_G1" & cell_type=="beta"').index.tolist()
               }
# Use PANGLAO terms
db='PANGLAO'
terms_idx=np.argwhere([t.startswith(db) for t in adata.uns['terms']]).ravel()
# Cell type scores
np.random.seed(0)
torch.manual_seed(0)
scores = model.latent_enrich(
    condition_dict, comparison="beta_control", directions=directions, adata=adata, 
    select_terms=terms_idx,n_perm=10000)

# Print out results
for ct in scores:
    print('beta cells: STZ-treated vs control')
    data=pd.DataFrame({'bf':scores[ct]['bf']},
                      index=[t.replace(db+'_','') for t in adata.uns['terms'][terms_idx]])
    data.index.name='term'
    data=data.query('abs(bf)>1').sort_values('bf',ascending=False,key=abs)
    max_rows=pd.options.display.max_rows
    pd.options.display.max_rows=data.shape[0]
    display(data.head(10))

# %% [markdown]
# Some of the enriched terms could be enriched due to ambience effects resulting from potential changes in cell type composition across samples. However, other might indicate true biological difference, such as enteroendocrine cells (umbrella term for different endocrine cells, including pancreatic endocrine cells) that could indicate transdiferentiation towards other pancreatic endocrine cell types, as was previously suggested.

# %%
# Plot endocrine score distribution in beta cells and unrelated cell type
cts=['beta','immune']
fig,ax=plt.subplots(len(cts),1,figsize=(6,4*len(cts)),sharex=True)
for idx,ct in enumerate(cts):
    sc.pl.violin(adata_latent_q[adata_latent_q.obs.cell_type==ct,:], 
                 keys='PANGLAO_ENTEROENDOCRINE_CELLS', groupby='design',rotation=90,
            #stripplot=False,
                 ax=ax[idx],show=False)
    ax[idx].set_title(ct)
    if idx!=0:
        ax[idx].set_ylabel('')
    else:
        ax[idx].set_ylabel('enteroendocrine score')
fig.tight_layout()

# %% [markdown]
# We observed marked increase in enteroendocrine score in STZ-treated beta cells compared to healthy control beta cells. We do not observe such a trend in an unrelated cell type (immune cells), which indicates that this is likely not due to cell type composition-related ambient effects.

# %% [markdown]
# #### Embedding separation
# Compare beta-cell embedding of healthy, T2D model, and treated T2D model to reference data.

# %%
# Compare query samples embedding to reference
# Randomly order cells in UMAP
rcParams['figure.figsize']=(6,6)
np.random.seed(0)
random_indices=np.random.permutation(list(range(adata_beta.shape[0])))
sc.pl.umap(adata_beta[random_indices,:],color='ref_querySample',s=10,
          save='_betaCells.pdf')

# %% [markdown]
# Healthy reference cells (named control) map mainly with reference cells (which are not T2 diabetic). Cells treated with STZ (T2 diabetes model) map largely to a different location. To support this observation from UMAP embedding we also compute PAGA of query samples and reference to more quantitatively asses the sample similarities.

# %%
# Compute PAGA
sc.tl.paga(adata_beta,groups='ref_querySample')

# %%
# Plot PAGA
fig,ax=plt.subplots(figsize=(3,3))
sc.pl.paga(adata_beta, color='ref_querySample',ax=ax,
           labels=['']*adata_beta.obs['ref_querySample'].nunique(),
           edge_width_scale=0.5, node_size_scale=0.3,fontsize=3,show=False,
          frameon=False)
handles = [Patch(facecolor=c) for c in adata_beta.uns['ref_querySample_colors']]
ax.legend(handles, adata_beta.obs['ref_querySample'].cat.categories,
           bbox_to_anchor=(1, 0.935), bbox_transform=plt.gcf().transFigure, frameon=False)
plt.savefig(path_fig+'betaCells_paga.pdf',dpi=300,bbox_inches='tight')

# %% [markdown]
# ### Differential gene-set scores
# Find Reactome gene sets with differential scores in control and STZ treated beta cells from STZ study.

# %%
# Compare WT and STZ beta cells from STZ study
condition_dict={
    'beta_STZ':adata.obs.query('study_sample=="STZ_G2" & cell_type=="beta"').index.tolist(),
    'beta_control':adata.obs.query('study_sample=="STZ_G1" & cell_type=="beta"').index.tolist()
               }
# Use REACTOME terms
db='REACTOME'
terms_idx=np.argwhere([t.startswith(db) for t in adata.uns['terms']]).ravel()
# Cell type scores
np.random.seed(0)
torch.manual_seed(0)
scores = model.latent_enrich(
    condition_dict, comparison="beta_control", directions=directions, adata=adata, 
    select_terms=terms_idx,n_perm=10000)

# Print out results
for ct in scores:
    print('beta cells: STZ-treated vs control')
    data=pd.DataFrame({'bf':scores[ct]['bf']},
                      index=[t.replace(db+'_','') for t in adata.uns['terms'][terms_idx]])
    data.index.name='term'
    data=data.sort_values('bf',ascending=False,key=abs)
    data.to_csv(path_res+'diabetes_reactomeEnrichment.tsv',sep='\t')
    data_enr=data.query('abs(bf)>1')
    max_rows=pd.options.display.max_rows
    pd.options.display.max_rows=data_enr.shape[0]
    display(data_enr)
    pd.options.display.max_rows=max_rows

# %% [markdown]
# There seem to be differences in: energy metabolism and protein synthesis, unfolded protein response, cell-matrix interactions and cell-cell (including immune) communication.

# %%
# Differential terms for plotting below
plot_terms=data_enr.index

# %% [markdown]
# We can extract genes that contribute the most to the activation of each term.

# %%
# Report genes contributing to each enriched term
weight_datas={}
term_id=dict(zip(plot_terms,range(len(plot_terms))))
for term in term_id.keys():
    term_idx=np.argwhere(adata.uns['terms']=='REACTOME_'+term)[0,0]
    # Only non-zero weight genes are reported
    weights=pd.DataFrame({
        'EID':adata.var_names.values,
        'gene_symbol':adata.var.gene_symbol.values,
        'weight':model.model.decoder.L0.expr_L.weight[:,term_idx].detach().numpy()})
    weights=weights.query('weight!=0')
    weights.sort_values('weight',key=abs,inplace=True,ascending=False)
    weight_datas[term]=weights
# Save non-zero weight genes of each term
writer = pd.ExcelWriter(path_res+'diabetes_reactomeEnriched_geneWeights.xlsx',
                        engine='xlsxwriter') 
for term,data in weight_datas.items():
    data.to_excel(writer, sheet_name=str(term_id[term]),index=False) 
# Save sheet name-term mapping
pd.DataFrame({'term':term_id.keys(),'sheet_name':term_id.values()}
            ).to_excel(writer, sheet_name='Sheet name terms',index=False) 
writer.save()

# %% [markdown]
# #### Enriched gene set overlap
# Compare genes across enriched terms to see overlap between gene sets.

# %%
# Get genes per term, use only genes used for integration (in adata)
terms_genes={}
with open(path_gmt+'c2.cp.reactome.v4.0_mouseEID.gmt','r') as f:
    for term_data in f.readlines():
        term_data=term_data.split()
        term=term_data[0]
        genes=set([gene for gene in term_data[2:] if gene in adata.var_names])
        terms_genes[term]=genes

# %% [markdown]
# Genes present in each enriched term:

# %%
# Which genes are present in each term
terms_genes_map=pd.DataFrame(index=plot_terms)
for term in plot_terms:
    genes=terms_genes[db+'_'+term]
    terms_genes_map.loc[term,genes]=1
terms_genes_map.fillna(0,inplace=True)

# %%
# Plot genes present in each term
sns.clustermap(terms_genes_map,xticklabels=False,method='ward')

# %% [markdown]
# The enriched gene sets do not have many overlapping terms. This is likely due to using high regularisation parameter alpha in integration for Reactome terms, leading to deactivation of redundant gene-set terms.

# %% [markdown]
# #### Distribution of term scores across conditions
# Enriched term scores distribution across all STZ-study samples. Most severe diabetes is observed in STZ-treated samples, improving most strongly upon treatment that include insulin. Control sample is healthy.

# %%
# Latent data of enriched terms in STZ study
latent_enr=pd.DataFrame(
    adata_beta[adata_beta.obs.study=='STZ',:].obsm['X_qtr_directed'],
    columns=[t.replace(db+'_','') for t in adata_beta.uns['terms']],
    index=adata_beta.obs_names[adata_beta.obs.study=='STZ'])[plot_terms]

# %%
# Parse latent data subset for sns plotting
swarmplot_df=latent_enr.stack().reset_index().rename({'level_1':'term',0:'score'},axis=1)
swarmplot_df['ref_querySample']=swarmplot_df.apply(
    lambda x: adata_beta.obs.at[x['index'],'ref_querySample'],axis=1)
swarmplot_df['ref_querySample']= pd.Categorical(swarmplot_df['ref_querySample'], 
                      categories=[c for c in list(colormap.keys()) 
                                  if c in swarmplot_df['ref_querySample'].unique()],
                      ordered=True)

# %%
# Plot distribution of term scores across STZ samples
a=sns.catplot(x="score", y="ref_querySample",
            row="term",orient="h",
            data=swarmplot_df, 
            kind="violin",inner=None,
            height=4, aspect=0.7, sharex=False,
            palette=[colormap[c] for c in swarmplot_df['ref_querySample'].cat.categories])
plt.savefig(path_fig+'diabetes_reactomeEnrichment.pdf',dpi=300,bbox_inches='tight')

# %% [markdown]
# The term about immune interactions between lymphoid and non-limphoid cells shows bimodal distribution - something that could not have been identified if we did not had per-cell scores obtained from expiMap Thus we also plot how the distributions look in other reference samples to identify which scores located in similar regions.

# %%
# Term distributions across samples
term= 'IMMUNOREGULATORY_INTERACTIONS_BETWEEN_A_LYMPHOID_AND_A_NON_LYMPHOID_CELL'
# Unused palette study+query samples
#cmap_study=dict(zip(adata_beta.obs['study'].cat.categories,adata_beta.uns['study_colors']))
#cmap_samples_q=dict(zip(adata_beta.obs['ref_querySample'].cat.categories,
#                        adata_beta.uns['ref_querySample_colors']))
#palette=[]
#for sample in adata_beta.obs['study_design'].cat.categories:
#    rq=adata_beta[adata_beta.obs['study_design']==sample].obs.ref_query.values[0]
#    if rq=='ref':
#        study=adata_beta[adata_beta.obs['study_design']==sample].obs.study.values[0]
#        palette.append(cmap_study[study])
#    else:
#        sample=adata_beta[adata_beta.obs['study_design']==sample].obs.ref_querySample.values[0]
#        palette.append(cmap_samples_q[sample])
a=sns.violinplot(
        x=pd.Series(
            adata_beta.obsm['X_qtr_directed'][:, 
            np.argwhere(adata_beta.uns['terms']=='REACTOME_'+term)[0][0]],
            name=term,index=adata_beta.obs_names), 
            y=adata_beta.obs["study_design"],
            sym='',inner=None,
            palette=[cmap[adata_beta.obs.query('study_design==@study_design')['study'].values[0]] 
                for study_design in adata_beta.obs['study_design'].cat.categories]
)
plt.savefig(path_fig+'samples_ImmunoregulatoryTerm.pdf',dpi=300,bbox_inches='tight')

# %% [markdown]
# #### Comparison of terms
# Comparing different terms can give us additional insights into term relationships and their usability for separating cell populations.

# %% [markdown]
# Unfolded protein response is one of the hallmarks of type 2 diabetes. It arises due to increase in insulin production that exceeds the protein processing capacity of the beta cells. Thus we here compare activity distribution of differentially active gene set terms associated with protein metabolism. This includes terms related to translation and protein folding, processing, and secretion.

# %%
# Plot terms comparison in beta cells
terms2=[
    'METABOLISM_OF_MRNA',
    'TRANSLATION',
    'PROTEIN_FOLDING',
    'MEMBRANE_TRAFFICKING',
    'ASPARAGINE_N_LINKED_GLYCOSYLATION',
]
t1='UNFOLDED_PROTEIN_RESPONSE'
# Randomise plotting order
np.random.seed(0)
# Sort random indices for ref to be on the bottom
indices=np.array(range(adata_beta.shape[0]))
random_indices=list(np.random.permutation(indices[adata_beta.obs.ref_query=='ref']))+\
                list(np.random.permutation(indices[adata_beta.obs.ref_query=='query']))
fig,axs=plt.subplots(1,len(terms2),figsize=(len(terms2)*4+2,4),sharey=False,sharex=True)
for idx,t2 in enumerate(terms2):
    terms_plot=[t1,t2]
    for term in terms_plot:
        adata_beta.obs[term]=adata_beta.obsm[
        'X_qtr_directed'][:, np.argwhere(adata_beta.uns['terms']=='REACTOME_'+term)[0][0]]
    ax=axs[idx]
    sns.scatterplot(x=adata_beta[random_indices,:].obs[terms_plot[0]],
                y=adata_beta[random_indices,:].obs[terms_plot[1]],
                hue=adata_beta[random_indices,:].obs['ref_querySample'],
                palette=dict(zip(adata_beta.obs['ref_querySample'].cat.categories,
                 adata_beta.uns['ref_querySample_colors'])),
                s=0.5,rasterized=True,ax=ax)
    corr=np.corrcoef(adata_beta.obs[terms_plot[0]].values.ravel(),
                     adata_beta.obs[terms_plot[1]].values.ravel())[0,1]
    ax.set_title('Correlation: %.2f'%corr)
    if idx!=len(terms2)-1:
        ax.get_legend().remove()
    else:
        ax.legend(bbox_to_anchor=(1.1, 1.05),frameon=False)
        ax.get_legend().get_frame().set_facecolor('none')
    adata_beta.obs.drop(terms_plot,axis=1,inplace=True)
fig.tight_layout()
plt.savefig(path_fig+'UPR_term_comparison.pdf',dpi=300,bbox_inches='tight')

# %% [markdown]
# It seems that some terms differentially active between healthy and STZ-treated query cells do not separate non-T2D diabetic reference cells from STZ treated query cells. ExpiMap thus better allows us to select T2D-specific gene sets as it enables direct comparison to the reference by using the batch-corrected latent space. Below we further check in which reference samples we observe increased mRNA metabolism without UPR response.
#
# Furthermore, we observe high correlation between UPR and asparagine N-linked glycosylation across all samples. This is an interesting finding as N-linked glycosylation changes were previously observed in diabetes, but it is not well known why they arise. As N-linked glycosylation may be associated with changes in cell communication we also plot the comparison between differentially active cell communication terms and asparagine N-linked glycosylation scores.

# %% [markdown]
# Comparison to asparagine N-linked glycosylation:

# %%
# Plot terms comparison in beta cells
terms2=[
    'NEUROTRANSMITTER_RECEPTOR_BINDING_AND_DOWNSTREAM_TRANSMISSION_IN_THE_POSTSYNAPTIC_CELL',
    'INTEGRIN_CELL_SURFACE_INTERACTIONS',
    'GLYCEROPHOSPHOLIPID_BIOSYNTHESIS',
    'PHOSPHOLIPID_METABOLISM',
    'COLLAGEN_FORMATION',
    'CHONDROITIN_SULFATE_DERMATAN_SULFATE_METABOLISM'
]
t1='ASPARAGINE_N_LINKED_GLYCOSYLATION'
np.random.seed(0)
# Sort random indices for ref to be on the bottom
indices=np.array(range(adata_beta.shape[0]))
random_indices=list(np.random.permutation(indices[adata_beta.obs.ref_query=='ref']))+\
                list(np.random.permutation(indices[adata_beta.obs.ref_query=='query']))
fig,axs=plt.subplots(1,len(terms2),figsize=(len(terms2)*4+6,4),sharey=False,sharex=True)
for idx,t2 in enumerate(terms2):
    terms_plot=[t1,t2]
    for term in terms_plot:
        adata_beta.obs[term]=adata_beta.obsm[
        'X_qtr_directed'][:, np.argwhere(adata_beta.uns['terms']=='REACTOME_'+term)[0][0]]
    ax=axs[idx]
    sns.scatterplot(x=adata_beta[random_indices,:].obs[terms_plot[0]],
                y=adata_beta[random_indices,:].obs[terms_plot[1]],
                hue=adata_beta[random_indices,:].obs['ref_querySample'],
                palette=dict(zip(adata_beta.obs['ref_querySample'].cat.categories,
                 adata_beta.uns['ref_querySample_colors'])),
                s=0.5,rasterized=True,ax=ax)
    corr=np.corrcoef(adata_beta.obs[terms_plot[0]].values.ravel(),
                     adata_beta.obs[terms_plot[1]].values.ravel())[0,1]
    ax.set_title('Correlation: %.2f'%corr)
    if idx!=len(terms2)-1:
        ax.get_legend().remove()
    else:
        ax.legend(bbox_to_anchor=(1.1, 1.05),frameon=False)
        ax.get_legend().get_frame().set_facecolor('none')
    adata_beta.obs.drop(terms_plot,axis=1,inplace=True)
fig.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                    wspace=0.3)
plt.savefig(path_fig+'NGly_communication_comparison.pdf',dpi=300,bbox_inches='tight')

# %%
# Plot terms comparison in beta cells
terms2=[
    'IMMUNOREGULATORY_INTERACTIONS_BETWEEN_A_LYMPHOID_AND_A_NON_LYMPHOID_CELL',
    'INNATE_IMMUNE_SYSTEM',
    'L1CAM_INTERACTIONS',
]
t1='ASPARAGINE_N_LINKED_GLYCOSYLATION'
np.random.seed(0)
# Sort random indices for ref to be on the bottom
indices=np.array(range(adata_beta.shape[0]))
random_indices=list(np.random.permutation(indices[adata_beta.obs.ref_query=='ref']))+\
                list(np.random.permutation(indices[adata_beta.obs.ref_query=='query']))
fig,axs=plt.subplots(1,len(terms2),figsize=(len(terms2)*4+4,4),sharey=False,sharex=True)
for idx,t2 in enumerate(terms2):
    terms_plot=[t1,t2]
    for term in terms_plot:
        adata_beta.obs[term]=adata_beta.obsm[
        'X_qtr_directed'][:, np.argwhere(adata_beta.uns['terms']=='REACTOME_'+term)[0][0]]
    ax=axs[idx]
    sns.scatterplot(x=adata_beta[random_indices,:].obs[terms_plot[0]],
                y=adata_beta[random_indices,:].obs[terms_plot[1]],
                hue=adata_beta[random_indices,:].obs['ref_querySample'],
                palette=dict(zip(adata_beta.obs['ref_querySample'].cat.categories,
                 adata_beta.uns['ref_querySample_colors'])),
                s=0.5,rasterized=True,ax=ax)
    corr=np.corrcoef(adata_beta.obs[terms_plot[0]].values.ravel(),
                     adata_beta.obs[terms_plot[1]].values.ravel())[0,1]
    ax.set_title('Correlation: %.2f'%corr)
    if idx!=len(terms2)-1:
        ax.get_legend().remove()
    else:
        ax.legend(bbox_to_anchor=(1.1, 1.05),frameon=False)
        ax.get_legend().get_frame().set_facecolor('none')
    adata_beta.obs.drop(terms_plot,axis=1,inplace=True)
fig.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                    wspace=0.3)
plt.savefig(path_fig+'NGly_immune_comparison.pdf',dpi=300,bbox_inches='tight')

# %% [markdown]
# Make subplot only of innate immune response and asparagine N-glycosylation comparison.

# %%
# Plot terms comparison in beta cells
t2= 'INNATE_IMMUNE_SYSTEM'
t1='ASPARAGINE_N_LINKED_GLYCOSYLATION'
np.random.seed(0)
# Sort random indices for ref to be on the bottom
indices=np.array(range(adata_beta.shape[0]))
random_indices=list(np.random.permutation(indices[adata_beta.obs.ref_query=='ref']))+\
                list(np.random.permutation(indices[adata_beta.obs.ref_query=='query']))
fig,ax=plt.subplots(1,1,figsize=(4,4),sharey=False,sharex=True)

terms_plot=[t1,t2]
for term in terms_plot:
    adata_beta.obs[term]=adata_beta.obsm[
    'X_qtr_directed'][:, np.argwhere(adata_beta.uns['terms']=='REACTOME_'+term)[0][0]]
sns.scatterplot(x=adata_beta[random_indices,:].obs[terms_plot[0]],
            y=adata_beta[random_indices,:].obs[terms_plot[1]],
            hue=adata_beta[random_indices,:].obs['ref_querySample'],
            palette=dict(zip(adata_beta.obs['ref_querySample'].cat.categories,
             adata_beta.uns['ref_querySample_colors'])),
            s=0.5,rasterized=True,ax=ax)
corr=np.corrcoef(adata_beta.obs[terms_plot[0]].values.ravel(),
                     adata_beta.obs[terms_plot[1]].values.ravel())[0,1]
ax.set_title('Correlation: %.2f'%corr)
ax.legend(bbox_to_anchor=(1.1, 1.05),frameon=False)
ax.get_legend().get_frame().set_facecolor('none')
adata_beta.obs.drop(terms_plot,axis=1,inplace=True)
fig.tight_layout()
plt.savefig(path_fig+'NGly_innateImmune_comparison.pdf',dpi=300,bbox_inches='tight')

# %% [markdown]
# Distribution of different terms differentially active in T2D model across all samples (including reference).

# %%
# Distribution of selected terms across all samples on beta cells
terms=['METABOLISM_OF_MRNA','UNFOLDED_PROTEIN_RESPONSE',
             'ASPARAGINE_N_LINKED_GLYCOSYLATION']
fig,ax=plt.subplots(1,len(terms),figsize=(5*len(terms),10),sharey=True)
for idx,term in enumerate(terms):
    cmap=dict(zip(adata_beta.obs['study'].cat.categories,adata_beta.uns['study_colors']))
    a=sns.boxplot(
        x=pd.Series(
            adata_beta.obsm['X_qtr_directed'][:, 
            np.argwhere(adata_beta.uns['terms']=='REACTOME_'+term)[0][0]],
            name=term,index=adata_beta.obs_names), 
            y=adata_beta.obs["study_design"],
            sym='',
            palette=[cmap[adata_beta.obs.query('study_design==@study_design')['study'].values[0]] 
                for study_design in adata_beta.obs['study_design'].cat.categories],ax=ax[idx])
    if idx!=0:
        ax[idx].set_ylabel('')
plt.savefig(path_fig+'samples_UprRelatedReactomeTerms.pdf',dpi=300,bbox_inches='tight')

# %% [markdown]
# Increased mRNA metabolism without proportionally increased UPR response is observed in some young NOD samples and samples treated with FoxO and arthemeter. All these conditions may be related to beta cell stress, but seemingly not through UPR.

# %% [markdown]
# #### Correlation of enriched terms
# Do terms change differently across STZ samples, thus being up/down regulated at different disease severities? For example, it would be possible that some terms show strongest change in severe T2D (e.g. between STZ and insulin treated samples), while others show change latter during healing course, between insluin-treated and healthy samples. If this was the case one would expect to see subgroups of terms with stronger enrichment.
#
# Compute correlation between terms using STZ-study beta cells from different conditions. We multiply correlation between terms with oposite direction of change (e.g. bf from enrichment analysis) with -1 to ensure that we do not get negative corelations for opositely oriented terms.

# %%
# Correlation between terms on STZ-beta cells, multiply by -1 if terms have oposite bf
term_correlation=pd.DataFrame(index=plot_terms,columns=plot_terms)
directions_dict=dict(zip(adata.uns['terms'],directions))
for ii in range(len(plot_terms)-1):
    for ij in range(ii+1,len(plot_terms)):
        corr=np.corrcoef(latent_enr.iloc[:,ii],latent_enr.iloc[:,ij])[0,1]
        t1=plot_terms[ii]
        t2=plot_terms[ij]
        if data.loc[t1,'bf']*data.loc[t2,'bf'] < 0:
            corr=corr*-1
        term_correlation.at[t1,t2]=corr
        term_correlation.at[t2,t1]=corr
for ii in range(len(plot_terms)):
    term_correlation.at[plot_terms[ii],plot_terms[ii]]=1

# %%
# Plot term correlation
g=sns.clustermap(term_correlation.astype('float'),cmap='coolwarm',vmin=-1,vmax=1,center=0)

# %% [markdown]
# We do not see any clear separation of terms into different groups.
