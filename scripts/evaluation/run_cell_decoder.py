import scanpy as sc
import argparse
import scarches as sca
import torch

parser = argparse.ArgumentParser()

parser.add_argument("adata_file")
parser.add_argument("batch_col")
parser.add_argument("--epochs", dest="epochs", type=int, default=None)
parser.add_argument("--seed", dest="seed", type=int, default=None)

args = parser.parse_args()

adata_file = args.adata_file
batch_col = args.batch_col

print(adata_file, batch_col)

if args.seed is not None:
    print('Setting torch seed to', args.seed)
    torch.manual_seed(args.seed)

adata = sc.read(adata_file)

sc.pp.pca(adata, n_comps=10)

sca.models.CellDecoder.setup_anndata(adata, batch_key=batch_col, layer='counts')

lrs = [0.2, 0.15, 0.1, 0.05]
use_pca = [False, True]

for with_pca in use_pca:
    for lr in lrs:
        cdec = sca.models.CellDecoder(
            adata,
#            use_batch_norm_decoder=False,
#            use_layer_norm_decoder=True,
#            n_layers=2
        )
        if with_pca:
            cdec.module.z_m.data = torch.tensor(adata.obsm['X_pca'].copy())
        cdec.train(max_epochs=600 if args.epochs is None else args.epochs,
                   batch_size=256, early_stopping=True, train_size=1.,
                   plan_kwargs=dict(kl_weight=1., weight_decay=1e-6, lr=lr))

        lat_name = f"X_cdec_lr_{lr}_pca_{with_pca}"
        if args.epochs is not None:
            lat_name += f'_epochs_{args.epochs}'

        adata.obsm[lat_name] = cdec.get_latent_representation()

adata.write(adata_file)
