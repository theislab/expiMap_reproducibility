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

while getopts ":a:k:r:u:b:s:p:d:l:h:e:n:o:S:" flag
do
    case "${flag}" in
        a) a=${OPTARG};;
        k) akl=${OPTARG};;
        r) ra=${OPTARG};;
        u) uh=${OPTARG};;
        b) b=${OPTARG};;
        s) sd=${OPTARG};;
        p) dp=${OPTARG};;
        d) d=${OPTARG};;
        l) lr=${OPTARG};;
        h) hls=${OPTARG};;
        e) es=${OPTARG};;
        n) nh=${OPTARG};;
        o) ne=${OPTARG};;
        S) sa=${OPTARG};;
    esac
done


echo Script params alpha $a alpha_kl $akl remove_ambient $ra use_hvg $uh batch $b subset_data $sd data_proportion $dp dir $d learning_rate $lr hidden_layer_sizes $hls early_stopping $es n_hvg $nh n_epochs $ne save $sa decoder_last_layer $dll
python  /storage/groups/ml01/code/karin.hrovatin/qtr_intercode_reproducibility-/pancreas/integration_gsPanglaoReactome_paramOptimisation.py -a "$a" -akl "$akl" -ra "$ra" -uh "$uh" -b "$b" -sd "$sd" -dp "$dp" -d "$d" -lr "$lr" -hls "$hls" -es $es -nh $nh -ne $ne -s $sa
