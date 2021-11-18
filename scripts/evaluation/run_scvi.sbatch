#!/bin/bash

#SBATCH -o /storage/groups/ml01/workspace/sergei.rybakov/data_integr/integration_scVI_%j.job
#SBATCH -e /storage/groups/ml01/workspace/sergei.rybakov/data_integr/integration_scVI_%j.job
#SBATCH -J integration_scVI
#SBATCH -c 4 #CPU cores required
#SBATCH --mem=60G #Memory required
#SBATCH -t 48:00:00 #Job runtime
#SBATCH --nice=10000 #Manual priority. Do not change this.
#SBATCH -p gpu_p
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1 #How many GPUs you request, use 1

source $HOME/.bashrc
source activate work-gpu

python  $HOME/integr/run_scvi.py /storage/groups/ml01/workspace/sergei.rybakov/data_integr/lung_travaglini_preproc.h5ad tech_sample
python  $HOME/integr/run_scvi.py /storage/groups/ml01/workspace/sergei.rybakov/data_integr/liver_popescu_preproc.h5ad orig.ident
python  $HOME/integr/run_scvi.py /storage/groups/ml01/workspace/sergei.rybakov/data_integr/colon_smilie_preproc.h5ad Source
python  $HOME/integr/run_scvi.py /storage/groups/ml01/workspace/sergei.rybakov/data_integr/blood_azimuth_preproc.h5ad donor
python  $HOME/integr/run_scvi.py /storage/groups/ml01/workspace/sergei.rybakov/data_integr/heart_scvi_preproc.h5ad cell_source
