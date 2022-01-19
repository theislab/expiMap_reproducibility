import scarches as sca
import scanpy as sc
import numpy as np
import pandas as pd

print('Reading and processing the data.')

adata = sc.read('kang_count.h5ad')
sca.add_annotations(adata, 'c2.cp.reactome.v4.0.symbols.gmt', min_genes=12)
adata._inplace_subset_var(adata.varm['I'].sum(1)>0)
adata.obs['study'] = 'Kang'
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    flavor="seurat_v3",
    subset=True)
select_terms = adata.varm['I'].sum(0)>12
adata.uns['terms'] = np.array(adata.uns['terms'])[select_terms].tolist()
adata.varm['I'] = adata.varm['I'][:, select_terms]
adata._inplace_subset_var(adata.varm['I'].sum(1)>0)

rm_terms = ['CYTOKINE_SIGNALING_IN_IMMUNE_S', 'INTERFERON_ALPHA_BETA_SIGNALIN',
            'ANTIVIRAL_MECHANISM_BY_IFN_STI', 'INTERFERON_GAMMA_SIGNALING',
            'IMMUNE_SYSTEM']
ix_f = []
for t in rm_terms:
    ix_f.append(adata.uns['terms'].index(t))
for i in ix_f:
    del adata.uns['terms'][i]
adata.varm['I'] = np.delete(adata.varm['I'], ix_f, axis=1)

print('Training the reference for selecting top genes.')

ref_soft = False

intr_cvae = sca.models.TRVAE(
    adata=adata,
    condition_key='study',
    hidden_layer_sizes=[256, 256, 256],
    use_mmd=False,
    recon_loss='nb',
    mask=adata.varm['I'].T,
    use_decoder_relu=False,
    n_ext_decoder=0,
    n_expand_encoder=0,
    soft_mask=ref_soft,
    use_hsic=False,
    hsic_one_vs_all=False
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
    alpha_l1= 0.5 if ref_soft else None,
    gamma_ext=None,
    gamma_epoch_anneal=None,
    alpha_kl=0.06,
    beta=None,
    weight_decay=0.,
    early_stopping_kwargs=early_stopping_kwargs,
    use_early_stopping=True,
    print_n_deactive=False,
    seed=2020
)

alphas_l1 = (0.4, 0.3, 0.2, 0.1, 0.06)
del_n_genes =(5, 10, 15)

ifn_idx = adata.uns['terms'].index('INTERFERON_SIGNALING')
ifn_scores = lambda m: m.model.decoder.L0.expr_L.weight.data[:, ifn_idx].abs().cpu().numpy()

top_genes = np.argsort(ifn_scores(intr_cvae))[::-1][:max(del_n_genes)]

ifn_nz = np.where(adata.varm['I'][:, ifn_idx].astype(bool))[0]

df = None

for n_g in del_n_genes:
    print('Delete n genes:', n_g)

    df_n_g = pd.DataFrame(index=[f'del_{n_g}'])

    I_del = adata.varm['I'].copy()
    del_g = top_genes[:n_g]
    I_del[del_g, ifn_idx] = 0.

    for alpha_l1 in alphas_l1:
        print('  use alpha_l1', alpha_l1)

        intr_cvae = sca.models.TRVAE(
            adata=adata,
            condition_key='study',
            hidden_layer_sizes=[256, 256, 256],
            use_mmd=False,
            recon_loss='nb',
            mask=I_del.T,
            use_decoder_relu=False,
            n_ext_decoder=0,
            n_expand_encoder=0,
            soft_mask=True,
            use_hsic=False,
            hsic_one_vs_all=False
        )

        intr_cvae.train(
            n_epochs=200,
            alpha_epoch_anneal=100,
            alpha=ALPHA,
            omega=OMEGA,
            alpha_l1=alpha_l1,
        #    alpha_l1_epoch_anneal=50,
            gamma_ext=None,
            gamma_epoch_anneal=None,
            alpha_kl=0.06,
            beta=None,
            weight_decay=0.,
            early_stopping_kwargs=early_stopping_kwargs,
            use_early_stopping=True,
            print_n_deactive=False,
            seed=2020
        )

        ifn_scores_l = ifn_scores(intr_cvae)

        for n in (20, 30, 50):
            ifn_top = np.argsort(ifn_scores_l)[-n:]
            in_ifn = np.isin(ifn_top, ifn_nz).sum()
            recov = np.isin(del_g, ifn_top).sum()

            print(f'    IFN in top {n}:', in_ifn)
            print(f'    Recovered in top {n}:', recov)

            df_n_g[f'al1_{alpha_l1}_recov_top{n}'] = recov
            df_n_g[f'al1_{alpha_l1}_in_ifn_top{n}'] = in_ifn

        scores_cond = intr_cvae.latent_enrich('condition', comparison="control",
                                              adata=adata, n_perm=7000)

        top_term_idx = np.argmax(np.abs(scores_cond['stimulated']['bf']))
        df_n_g[f'al1_{alpha_l1}_top_term'] = adata.uns['terms'][top_term_idx]

        df_n_g[f'al1_{alpha_l1}_ifn_bf'] = scores_cond['stimulated']['bf'][ifn_idx]

    df = df_n_g if df is None else df.append(df_n_g)

df.to_csv(f'gene_recovery_kang_ref_soft_{ref_soft}.csv')
