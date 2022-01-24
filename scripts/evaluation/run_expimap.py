import scanpy as sc
import scarches as sca
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("adata_file")
parser.add_argument("batch_col")

args = parser.parse_args()

adata_file = args.adata_file
batch_col = args.batch_col

print(adata_file, batch_col)

adata = sc.read(adata_file)

adata_train = adata.copy()
adata_train.X = adata.layers['counts'].copy()

early_stopping_kwargs = {
    "early_stopping_metric": "val_unweighted_loss",
    "threshold": 0,
    "patience": 50,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}

alphas_kl = [0.5, 0.1, 0.05, 0.01, 0.005]
for i, alpha_kl in enumerate(alphas_kl):
    intr_cvae = sca.models.TRVAE(
        adata=adata_train,
        condition_key=batch_col,
        hidden_layer_sizes=[256, 256, 256],
        use_mmd=False,
        recon_loss='nb',
        mask=adata.varm['I'].T,
        use_decoder_relu=False
    )

    intr_cvae.train(
        n_epochs=500,
        alpha_epoch_anneal=100,
        alpha=0.7,
        omega=None,
        alpha_kl=alpha_kl,
        weight_decay=0.,
        early_stopping_kwargs=early_stopping_kwargs,
        use_early_stopping=True,
        seed=2020
    )

    active = intr_cvae.model.decoder.nonzero_terms()
    adata.obsm["X_expimap_"+str(i)] = intr_cvae.get_latent(mean=False)[:, active]

adata.write(adata_file)
