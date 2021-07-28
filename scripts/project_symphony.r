library(symphony)
library(anndata)

project_symphony <- function(adata_file, batch_col, query_name, dim=20)
{
    ad <- read_h5ad(adata_file)
    
    q_mask = ad$obs[[batch_col]] == query_name
    
    ref_X = t(ad$X[!q_mask,])
    query_X = t(ad$X[q_mask,])
    
    ref_meta = ad$obs[!q_mask,][batch_col]
    query_meta = ad$obs[q_mask,][batch_col]
    
    reference = buildReference(
        ref_X,
        ref_meta,
        vars = c(batch_col),
        K = 100,
        verbose = TRUE,
        do_umap = FALSE,
        do_normalize = TRUE,
        vargenes_method = 'vst',
        topn = 2000,
        d = dim
    )
    
    query = mapQuery(
        query_X, 
        query_meta, 
        reference, 
        do_normalize = TRUE,
        do_umap = FALSE
    )
    
    latent <- t(cbind(reference[["Z_corr"]], query[["Z"]]))
    ad$obsm[["X_symphony"]] <- latent[ad$obs_names,]
    
    ad$write_h5ad(adata_file)
    
}