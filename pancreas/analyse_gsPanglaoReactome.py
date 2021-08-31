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
    display(data.sort_values('bf',ascending=False).iloc[:10])

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
for term, cts in terms_ct.items():
    term_idx=np.argwhere(adata.uns['terms']==term)[0][0]
    adata.obs[term]=adata.obsm['X_qtr_directed'][:,term_idx]                                     
    p=sc.pl.violin(adata, keys=term, groupby='cell_type',stripplot=False,rotation=90,show=False)
    # Mark ct median
    ct_median=adata.obs.query('cell_type in @cts')[term].median()    
    p.axhline(ct_median,c='k')
    # Mark ct
    ct_idxs=[np.argwhere(adata.obs.cell_type.cat.categories.values==cti)[0,0] for cti in cts]
    for ct_idx in ct_idxs:
        p.axvline(ct_idx,c='r')
    # Mark related cell types
    # Fix for stellate cells as doublets do not have subtype info
    if term=='PANGLAO_PANCREATIC_STELLATE_CELLS':
        cts.append('stellate')
    related_cts=[ct for ct in adata.obs.cell_type.unique()
                if any([c in ct for c in cts]) and ct not in cts]
    ct_idxs=[np.argwhere(adata.obs.cell_type.cat.categories.values==cti)[0,0] 
             for cti in related_cts]
    for ct_idx in ct_idxs:
        p.axvline(ct_idx,c='orange')
    adata.obs.drop(term,axis=1,inplace=True)

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
           cmap='coolwarm',vcenter=0)
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

# %%
# Prediction certainty
rcParams['figure.figsize']=(6,6)
sc.pl.umap(adata_latent_q,  color=["evaluation",'uncertainty'], frameon=False,  size=10)

# %% [markdown]
# Cell clusters with low prediction probability (e.g. immune) can be annotated based on cell type term scores, either with enrichment of cell clusters (as shown above) or with plots of score distributions of cell types expected to occur in the sample tissue.

# %%
plot_terms=list(terms_ct.keys())
rcParams['figure.figsize']=(6,6)
# Set colormap values for plotting, no min/max
sc.pl.umap(adata_latent_q,color=plot_terms,s=10,cmap='coolwarm',vcenter=0)

# %% [markdown]
# This helps us to identify some cell types that were not predicted, such as immune cells (absent from reference) and acinar cells (low number of cells).

# %% [markdown]
# ## Beta cell function - T2D model vs healthy
# Compare beta cells from healthy (control) and T2D-model (STZ-treated) samples from STZ study to find gene sets different in T2D.

# %%
# Make beta cell adata
adata_beta=adata[adata.obs.cell_type=='beta',:].copy()
sc.pp.neighbors(adata_beta, use_rep='X_qtr_directed')
sc.tl.umap(adata_beta)

# %% [markdown]
# ### Healthy and diseased beta cell embedding
# Compare beta-cell embedding of healthy, T2D model, and treated T2D model to reference data.

# %%
# Compare query samples embedding to reference
# Report cell as belonging to ref or one of the query designs
adata_beta.obs['ref_querySample']=pd.Series(
    [rq if rq=='ref' else design
    for rq, design in zip(adata_beta.obs.ref_query,adata_beta.obs.design)],
    dtype='category',index=adata_beta.obs_names)
# Set colors 
colormap={'STZ':'cyan',
          'STZ_GLP-1':'tab:purple','STZ_estrogen':'tab:pink','STZ_GLP-1_estrogen':'pink',
         'STZ_GLP-1_estrogen+insulin':'orange','STZ_insulin':'gold',
         'control':'yellowgreen', 'ref':'k'}
adata_beta.uns['ref_querySample_colors']=[
    colormap[cat] for cat in adata_beta.obs['ref_querySample'].cat.categories]
# Randomly order cells in UMAP
np.random.seed(0)
random_indices=np.random.permutation(list(range(adata_beta.shape[0])))
sc.pl.umap(adata_beta[random_indices,:],color='ref_querySample',s=10)

# %% [markdown]
# Healthy reference cells (named control) map mainly with reference cells (which are not T2 diabetic). Cells treated with STZ (T2 diabetes model) map largely to a different location.

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
    data=data.query('abs(bf)>1').sort_values('bf',ascending=False,key=abs)
    max_rows=pd.options.display.max_rows
    pd.options.display.max_rows=data.shape[0]
    def style_pos_neg(v):
        if v>0:
            return 'color:black;background-color:#cc6868;'
        elif v<0:
            return 'color:black;background-color:#6890cc;'
        else:
            return 'color:black;background-color:white;'
    def style_positive(v, props=''):
        return props if v < 0 else None
    display(data.style.applymap(style_pos_neg))
    pd.options.display.max_rows=max_rows

# %% [markdown]
# There seem to be differences in: energy metabolism and protein synthesis, unfolded protein response, cell-cell and cell-matrix interactions (including immune communication).

# %%
# Differential terms for plotting below
plot_terms=data.index

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
# Overlap between enriched term genes as ratio of smaller gene set. Bf direction is marked with edge colors as above in enrichment table.

# %%
# Overlap between terms as ratio of smaller gene set
terms_overlap=pd.DataFrame()
for i in range(len(plot_terms)-1):
    for j in range(i+1,len(plot_terms)):
        t1=plot_terms[i]
        t2=plot_terms[j]
        g1=terms_genes[db+'_'+t1]
        g2=terms_genes[db+'_'+t2]
        metric=len(g1&g2)/min([len(g1),len(g2)])
        terms_overlap.at[t1,t2]=metric
        terms_overlap.at[t2,t1]=metric
terms_overlap.fillna(1,inplace=True)        

# %%
# Colors for terms - red if positive, else blue
term_direction_colors=pd.Series({term:'#cc6868' if term_data['bf']>0 else '#6890cc'
                       for term,term_data in data.iterrows()})

# %%
# Visualise overlap between terms
sns.clustermap(terms_overlap,row_colors=term_direction_colors,col_colors=term_direction_colors)

# %% [markdown]
# The enriched gene sets do not have many overlapping terms. This is likely due to using high regularisation parameter alpha in integration for Reactome terms, leading to deactivation of redundant gene-set terms.

# %% [markdown]
# TODO: Some terms with high overlap do not have same enrichment direction.

# %% [markdown]
# #### Term genes expression across samples
# Compare gene-term gene means in the two conditions to check if score direction is correct. TODO remove/Unused as gene-wise means in one condition are more often higher than in the other condition.

# %%
# Normalised expression of genes in both conditions, using only genes used for integration
adata_beta_g1g2=adata_beta[adata_beta.obs.study_sample.isin(['STZ_G1','STZ_G2']),:].copy()
sc.pp.normalize_total(adata_beta_g1g2, target_sum=1e6, exclude_highly_expressed=True) #TODO change this

m1=adata_beta_g1g2[(adata_beta_g1g2.obs.study_sample=='STZ_G1').values &
              (adata_beta_g1g2.obs.cell_type=='beta').values,:].X.mean(axis=0)
m2=adata_beta_g1g2[(adata_beta_g1g2.obs.study_sample=='STZ_G2').values &
              (adata_beta_g1g2.obs.cell_type=='beta').values,:].X.mean(axis=0)
print('Ratio of genes with mean G1>G2:',sum(m1>m2)/m1.shape[0])

adata_temp=sc.pp.log1p(adata_beta_g1g2,copy=True)

# Differences between gene-wise means get biased after log transform
m1=adata_temp[(adata_temp.obs.study_sample=='STZ_G1').values &
              (adata_temp.obs.cell_type=='beta').values,:].X.mean(axis=0)
m2=adata_temp[(adata_temp.obs.study_sample=='STZ_G2').values &
              (adata_temp.obs.cell_type=='beta').values,:].X.mean(axis=0)
print('Ratio of genes with mean G1>G2 after log transformation:',sum(m1>m2)/m1.shape[0])
del adata_temp

# %%
# Normalised expression computed on all expressed genes
adata_full=sc.read('/storage/groups/ml01/workspace/karin.hrovatin/data/pancreas/scRNA/qtr/adata_annotated.h5ad')

m1=np.array(adata_full[(adata_full.obs.study_sample=='STZ_G1').values &
              (adata_full.obs.cell_type=='beta').values,:].X.mean(axis=0)).ravel()
m2=np.array(adata_full[(adata_full.obs.study_sample=='STZ_G2').values &
              (adata_full.obs.cell_type=='beta').values,:].X.mean(axis=0)).ravel()
print('Ratio of genes with mean G1>G2:',
      sum(m1>m2)/m1.shape[0])

# %%
# Plot term genes expression in both conditions
for term in plot_terms:
    #print(term,'bf:',data.at[term,'bf'])
    genes=list(terms_genes[db+'_'+term])
    #sc.pl.matrixplot(
    #    adata_beta_g1g2, 
    #    var_names=genes, 
    #    groupby='design', use_raw=False,standard_scale='var')
    m1=adata_beta_g1g2[(adata_beta_g1g2.obs.study_sample=='STZ_G1').values &
              (adata_beta_g1g2.obs.cell_type=='beta').values,genes].X.mean(axis=0)
    m2=adata_beta_g1g2[(adata_beta_g1g2.obs.study_sample=='STZ_G2').values &
              (adata_beta_g1g2.obs.cell_type=='beta').values,genes].X.mean(axis=0)
    lm_ratios=np.log2((m2+1)/(m1+1))
    fig,ax=plt.subplots(figsize=(3,3))
    plt.hist(lm_ratios.tolist())
    plt.title(term+' bf: '+str(data.at[term,'bf']))
    plt.xlabel('log2(mean_STZ+1/mean_control+1)')
    plt.axvline(0,c='r')
    plt.grid(b=None)
    display(fig)
    plt.close()
    #m_lm_ratio=np.mean(lm_ratios)
    #print('Mean log2 ratio of group-wise means mean(log2(m_STZ+1/m_control+1)):',m_lm_ratio)

# %% [markdown]
# TODO: Distribution of term-associated genes log ratios of means in STZ vs control condition do not match bf direction.

# %%
# How many genes present in multiple active terms have both positive and negative weights?
print('Ratio of genes present in multiple active terms that have weights of different directions across terms',
      # has positive weight
      ((model.model.decoder.L0.expr_L.weight.detach().numpy()>0).any(axis=1) & 
       # has negative weight
      (model.model.decoder.L0.expr_L.weight.detach().numpy()<0).any(axis=1) ).sum() / 
      # N terms present in >1 active terms
      ((model.model.decoder.L0.expr_L.weight.detach().numpy()!=0).sum(axis=1)>1).sum()) 

# %% [markdown]
# TODO what to do with this? Genes have different weights across terms.

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
swarmplot_df['term']= pd.Categorical(swarmplot_df['term'], 
                      categories=g.data2d.index,
                      ordered=True)

# %%
# Plot distribution of term scores across STZ samples
sns.catplot(x="score", y="ref_querySample",
            row="term",orient="h",
            data=swarmplot_df, 
            kind="violin",inner=None,
            height=4, aspect=0.7, sharex=False,
            palette=[colormap[c] for c in swarmplot_df['ref_querySample'].cat.categories])

# %% [markdown]
# #### Correlation of enriched terms
# Do terms change differently across STZ samples, thus being up/down regulated at different disease severities? For example, it would be possible that some terms show strongest change in severe T2D (e.g. between STZ and insulin treated samples), while others show change latter during healing course, between insluin-treated and healthy samples. If this was the case one would expect to see subgroups of terms with stronger enrichment.
#
# Compute correlation between terms using STZ-study beta cells from different conditions.
#
# TODO remove as no interesting results

# %%
# Correlation between terms on STZ-beta cells
term_correlation=pd.DataFrame(index=plot_terms,columns=plot_terms)
for ii in range(len(plot_terms)-1):
    for ij in range(ii+1,len(plot_terms)):
        corr=np.corrcoef(latent_enr.iloc[:,ii],latent_enr.iloc[:,ij])[0,1]
        t1=plot_terms[ii]
        t2=plot_terms[ij]
        term_correlation.at[t1,t2]=corr
        term_correlation.at[t2,t1]=corr
for ii in range(len(plot_terms)):
    term_correlation.at[plot_terms[ii],plot_terms[ii]]=1

# %%
# Plot term correlation
g=sns.clustermap(term_correlation.astype('float'),cmap='coolwarm',vmin=-1,vmax=1,center=0)

# %% [markdown]
# Despite separation on up/down regulated terms there seems to be no subclusters within up/down regulated terms.

# %% [markdown]
# Same as above, but multiplying correlation of terms with oposite bf with -1.

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
