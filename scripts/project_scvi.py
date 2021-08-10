import scanpy as sc
import scvi

# .X should have unnormalized counts.
def project_scvi(adata_file, batch_col, query_names, dim=10):

    adata = sc.read(adata_file)

    q_mask = adata.obs[batch_col].isin(query_names)
    query = adata[q_mask].copy()
    ref = adata[~q_mask].copy()

    print("Integrating reference:")
    print(ref.obs[batch_col].unique())

    scvi.data.setup_anndata(ref, batch_key=batch_col)

    vae = scvi.model.SCVI(
        ref,
        n_layers=2,
        n_latent=dim,
        encode_covariates=True,
        deeply_inject_covariates=False,
        use_layer_norm="both",
        use_batch_norm="none",
    )

    vae.train(max_epochs=500, early_stopping=True)

    print("Projecting query:")
    print(query.obs[batch_col].unique())

    vae_q = sca.models.SCVI.load_query_data(query, vae, freeze_dropout=True)

    vae_q.train(max_epochs=500, early_stopping=True, plan_kwargs=dict(weight_decay=0.0))

    adata.obsm["X_scvi"] = vae_q.get_latent_representation(adata)

    adata.write(adata_file)
