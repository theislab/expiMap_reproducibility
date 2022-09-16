import scanpy as sc
import numpy as np
import scarches as sca
import pickle

ALPHA = 0.7
OMEGA = None
hsic = True
gamma_ext = 0.7

early_stopping_kwargs = {
    "early_stopping_metric": "val_unweighted_loss", # val_unweighted_loss
    "threshold": 0,
    "patience": 50,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}

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
rm_terms = ['INTERFERON_SIGNALING', 'INTERFERON_ALPHA_BETA_SIGNALIN',
            'CYTOKINE_SIGNALING_IN_IMMUNE_S', 'ANTIVIRAL_MECHANISM_BY_IFN_STI']
ix_f = []
for t in rm_terms:
    ix_f.append(adata.uns['terms'].index(t))
ifn_mask = adata.varm['I'][:, ix_f[0]][:, None].copy()
for i in ix_f:
    del adata.uns['terms'][i]
adata.varm['I'] = np.delete(adata.varm['I'], ix_f, axis=1)
adata._inplace_subset_var(adata.varm['I'].sum(1)>0)

kang = sc.read('./kang_count.h5ad')[:, adata.var_names].copy()
kang.obs['study'] = 'Kang'

ct_groups = [["CD16+ Monocytes"], ["CD14+ Monocytes"],
             ["CD4+ T cells", "CD8+ T cells"], ["Monocyte-derived dendritic cells"],
             ["CD20+ B cells", "CD10+ B cells"], ["NKT cells", "NK cells"]]

ct_query = [["CD16 Mono"], ["CD14 Mono"], ["CD4 T", "CD8 T", "T"],
            ["DC"], ["B"], ["NK"]]

ct_names = ["CD16_Mono", "CD14_Mono", "T", "DC", "B", "NK"]

stab = {}

for i, ct_group in enumerate(ct_groups):
    ct_name = ct_names[i]
    print(ct_name)

    stab[ct_name] = 0

    adata_noct = adata[~adata.obs.final_annotation.isin(ct_group)].copy()

    intr_cvae = sca.models.TRVAE(
        adata=adata_noct,
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

    kang.obs[ct_name] = "not_ct"
    for c in range(kang.n_obs):
        kang.obs[ct_name][c] = ct_name if np.isin(kang.obs['cell_type'][c], ct_query[i]) else 'not_ct'
    print(kang.obs[ct_name].value_counts())

    for j in range(20):
        print(f"ct {ct_name} iter {j}")

        q_intr_cvae = sca.models.TRVAE.load_query_data(
            kang,
            intr_cvae,
            unfreeze_ext=True,
            new_n_ext_decoder=10,
            new_n_expand_encoder=10
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
            alpha_l1=0.9,
            gamma_ext=gamma_ext,
            gamma_epoch_anneal=50,
            beta=3.,
            seed=2020,
            use_early_stopping=False,
            print_n_deactive=False
        )

        n_ref_terms = len(adata.uns['terms'])
        idx = [n_ref_terms + i for i in range(q_intr_cvae.model.n_ext_decoder)]

        scores_ct = q_intr_cvae.latent_enrich(
            ct_name,
            comparison="not_ct",
            adata=kang,
            n_perm=12000,
            exact=True
        )

        top_ct = np.argsort(np.abs(scores_ct[ct_name]["bf"]))[-1:]
        ct_enriched = any(np.isin(top_ct, idx)) and all(top_ct >= 2.3)
        print("ct enriched:", ct_enriched)
        if ct_enriched:
            stab[ct_name] += 1

pickle.dump(stab, open(f"./stab_ct_{gamma_ext}.pkl",'wb'))
