#!/bin/bash

#SBATCH -o /storage/groups/ml01/workspace/karin.hrovatin/slurm_integrationQTR_%j.job
#SBATCH -e /storage/groups/ml01/workspace/karin.hrovatin/slurm_integrationQTR_%j.job
#SBATCH -J integrationQTR
#SBATCH -c 4 #CPU cores required
#SBATCH --mem=50G #Memory required
#SBATCH -t 15:00:00 #Job runtime
#SBATCH --nice=10000 #Manual priority. Do not change this.
#SBATCH -p gpu_p
#SBATCH -q gpu
#SBATCH --gres=gpu:1 # N GPUs
#SBATCH --exclude=icb-gpusrv0[3-8],gpusrv[09,10],icb-gpusrv0[1-2],supergpu05 #Required for tf and pytorch as some nodes can not run it, also exclude interactive gpu nodes

source $HOME/.bashrc
conda activate qtr

python /storage/groups/ml01/code/karin.hrovatin/qtr_intercode_reproducibility-/pancreas/expiMap_integration_gsPanglaoReactome_paramOptimisation.py $1
