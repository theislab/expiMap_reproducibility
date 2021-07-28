#!/bin/bash

#SBATCH -o /storage/groups/ml01/workspace/karin.hrovatin/slurm_integrationEvalQTR_%j.job
#SBATCH -e /storage/groups/ml01/workspace/karin.hrovatin/slurm_integrationEvalQTR_%j.job
#SBATCH -J integrationEvalSpecies
#SBATCH -c 1 #CPU cores required
#SBATCH --mem=10G #Memory required, TODO put back to 350
#SBATCH -t 00:30:00 #Job runtime
#SBATCH --nice=10 #Manual priority. Do not change this.
#SBATCH -p cpu_p

source $HOME/.bashrc
conda activate scib2
# $1 - script file
# $2 - file path
# $3 - integrated obsm
# $4 - batch
# $5 - cell type
echo $1 $2 $3 $4 $5
python $1 -if $2 -ie $3 -b $4 -ct $5