import scanpy as sc
import numpy as np
import scarches as sca
import pickle

adata = sc.read('./pbmc_vars_sb.h5ad')
adata = adata[adata.obs['study']!='Villani'].copy()
adata.X = adata.layers["counts"].copy()
sca.add_annotations(adata, './c2.cp.reactome.v4.0.symbols.gmt', min_genes=12)
adata._inplace_subset_var(adata.varm['I'].sum(1)>0)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    batch_key="batch",
    subset=True)
select_terms = adata.varm['I'].sum(0)>12
adata.uns['terms'] = np.array(adata.uns['terms'])[select_terms].tolist()
adata.varm['I'] = adata.varm['I'][:, select_terms]
adata._inplace_subset_var(adata.varm['I'].sum(1)>0)
adata.X = adata.layers["counts"].copy()
rm_terms = ['INTERFERON_SIGNALING',
            'INTERFERON_ALPHA_BETA_SIGNALIN',
            "SIGNALING_BY_THE_B_CELL_RECEPT", "MHC_CLASS_II_ANTIGEN_PRESENTAT",
            'CYTOKINE_SIGNALING_IN_IMMUNE_S', 'ANTIVIRAL_MECHANISM_BY_IFN_STI']
ix_f = []
for t in rm_terms:
    ix_f.append(adata.uns['terms'].index(t))
ifn_mask = adata.varm['I'][:, ix_f[0]][:, None].copy()
query_mask = adata.varm['I'][:, ix_f[2]][:, None].copy()
for i in ix_f:
    del adata.uns['terms'][i]
adata.varm['I'] = np.delete(adata.varm['I'], ix_f, axis=1)
adata._inplace_subset_var(adata.varm['I'].sum(1)>0)
rm_b = ["CD20+ B cells", "CD10+ B cells"]
adata = adata[~adata.obs.final_annotation.isin(rm_b)].copy()


intr_cvae = sca.models.TRVAE(
    adata=adata,
    condition_key='study',
    hidden_layer_sizes=[300, 300, 300],
    use_mmd=False,
    recon_loss='nb',
    mask=adata.varm['I'].T,
    use_decoder_relu=False,
    n_ext_decoder=0, # add additional unannotated terms in decoder
    n_expand_encoder=0, # same for encoder, should be the same number
    soft_mask=False, # use soft mask
    use_hsic=False # use hsic
)

ALPHA = 0.7
OMEGA = None

early_stopping_kwargs = {
    "early_stopping_metric": "val_unweighted_loss", # val_unweighted_loss
    "threshold": 0,
    "patience": 50,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}
intr_cvae.train(
    n_epochs=200,
    alpha_epoch_anneal=100,
    alpha=ALPHA,
    omega=OMEGA,
    alpha_kl=0.5,
    weight_decay=0.,
    early_stopping_kwargs=early_stopping_kwargs,
    use_early_stopping=True,
    print_n_deactive=False,
    seed=2020
)

kang = sc.read('./kang_count.h5ad')[:, adata.var_names].copy()
kang.obs['study'] = 'Kang'

fractions = (1., 0.7, 0.5, 0.3, 0.1)

hsic = True

stab = dict(condition={})

stim_idx = np.where(kang.obs.condition == "stimulated")[0]
ctrl_idx = np.where(kang.obs.condition == "control")[0]

for frac in fractions:
    if frac == 1.:
        kang_tr = kang.copy()
    else:
        size = int(frac * len(stim_idx))
        frac_idx = np.concatenate((ctrl_idx, np.random.choice(stim_idx, size, replace=False)))
        kang_tr = kang[frac_idx].copy()

    stab["condition"][str(frac)] = 0

    for i in range(20):
        print(f"frac {frac} iter {i}")

        q_intr_cvae = sca.models.TRVAE.load_query_data(
            kang_tr,
            intr_cvae,
            unfreeze_ext=True,
            new_n_ext_decoder=3,
            new_n_ext_m_decoder=1,
            new_n_expand_encoder=4,
            new_ext_mask=query_mask.T,
            new_soft_ext_mask=True
        )

        q_intr_cvae.use_hsic_ = hsic
        q_intr_cvae.model.use_hsic = hsic
        q_intr_cvae.hsic_one_vs_all_= hsic
        q_intr_cvae.model.hsic_one_vs_all = hsic

        q_intr_cvae.train(
            n_epochs=150,
            alpha_epoch_anneal=50,
            alpha_kl=0.1,
            weight_decay=0.,
            alpha_l1=1.,
            gamma_ext=0.7,
            gamma_epoch_anneal=50,
            beta=3.,
            seed=2020,
            use_early_stopping=False,
            print_n_deactive=False
        )

        n_ref_terms = len(adata.uns['terms'])
        idx = [n_ref_terms + i for i in range(q_intr_cvae.model.n_ext_m_decoder + q_intr_cvae.model.n_ext_decoder)]

        scores_cond = q_intr_cvae.latent_enrich(
            'condition',
            comparison="control",
            adata=kang_tr,
            n_perm=12000,
            exact=True
        )

        cond_sorted = np.argsort(np.abs(scores_cond["stimulated"]["bf"]))
        top_cond = cond_sorted[-1:]
        cond_enriched = any(np.isin(top_cond, idx[1:])) and all(top_cond >= 2.3)
        print("cond enriched:", cond_enriched)
        if cond_enriched:
            stab["condition"][str(frac)] += 1

pickle.dump(stab, open("./stab_cond.pkl",'wb'))
