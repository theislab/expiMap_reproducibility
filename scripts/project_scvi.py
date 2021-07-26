import scanpy as sc
import scvi


def project_scvi(adata_file, batch_col, query_name, dim=10):
    
    adata = sc.read(adata_file)
    
    mask_query = adata.obs[batch_col] == query_name
    query = adata[mask_query].copy()
    ref = adata[~mask_query].copy()
    
    scvi.data.setup_anndata(ref, batch_key="study")

    vae = scvi.model.SCVI(
        adata,
        n_layers=2,
        n_latent=dim,
        encode_covariates=True,
        deeply_inject_covariates=False,
        use_layer_norm="both",
        use_batch_norm="none",
    )
    
    vae.train(max_epochs=500, early_stopping=True)
    
    vae_q = sca.models.SCVI.load_query_data(query, vae, freeze_dropout=True)
    
    vae_q.train(max_epochs=500, early_stopping=True, plan_kwargs=dict(weight_decay=0.0))
    
    adata.obsm["X_scvi"] = vae_q.get_latent_representation(adata)
    
    adata.write(adata_file)