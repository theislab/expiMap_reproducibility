#!/bin/bash

#SBATCH -o /storage/groups/ml01/workspace/sergei.rybakov/data_integr/integration_metrics_%j.job
#SBATCH -e /storage/groups/ml01/workspace/sergei.rybakov/data_integr/integration_metrics_%j.job
#SBATCH -J integration_metrics
#SBATCH -c 4 #CPU cores required
#SBATCH --mem=60G #Memory required
#SBATCH -t 48:00:00 #Job runtime
#SBATCH --nice=10000 #Manual priority. Do not change this.
#SBATCH -p cpu_p

source $HOME/.bashrc
source activate work-r

file_path=/storage/groups/ml01/workspace/sergei.rybakov/data_integr
script_path=$HOME/qtr_intercode_reproducibility-/scripts/integration_benchmarks

python $script_path/run_metrics.py -if $file_path/blood_azimuth_preproc.h5ad -ie X_scvi -b donor -ct celltype.l2
for i in 0 1 2 3 4
do
  python $script_path/run_metrics.py -if $file_path/blood_azimuth_preproc.h5ad -ie X_expimap_$i -b donor -ct celltype.l2
done

python $script_path/run_metrics.py -if $file_path/heart_scvi_preproc.h5ad -ie X_scvi -b cell_source -ct cell_type
for i in 0 1 2 3 4
do
  python $script_path/run_metrics.py -if $file_path/heart_scvi_preproc.h5ad -ie X_expimap_$i -b cell_source -ct cell_type
done

python $script_path/run_metrics.py -if $file_path/lung_travaglini_preproc.h5ad -ie X_scvi -b tech_sample -ct cell_type
for i in 0 1 2 3 4
do
  python $script_path/run_metrics.py -if $file_path/lung_travaglini_preproc.h5ad -ie X_expimap_$i -b tech_sample -ct cell_type
done

python $script_path/run_metrics.py -if $file_path/colon_smilie_preproc.h5ad -ie X_scvi -b Source -ct CellType
for i in 0 1 2 3 4
do
  python $script_path/run_metrics.py -if $file_path/colon_smilie_preproc.h5ad -ie X_expimap_$i -b Source -ct CellType
done

python $script_path/run_metrics.py -if $file_path/liver_popescu_preproc.h5ad -ie X_scvi -b orig.ident -ct cell.labels
for i in 0 1 2 3 4
do
  python $script_path/run_metrics.py -if $file_path/liver_popescu_preproc.h5ad -ie X_expimap_$i -b orig.ident -ct cell.labels
done
