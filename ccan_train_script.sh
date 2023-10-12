#!/bin/bash
#SBATCH -A def-kgroling
#SBATCH --array 1-50%3
#SBATCH --time 0-01:30:00
#SBARCH --n-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=v100:1

# Set the remote port and server
REMOTEHOST=34.105.9.156
REMOTEPORT=3306

# Set the local params
LOCALHOST=localhost
for ((i=0; i<10; ++i)); do
  LOCALPORT=$(shuf -i 1024-65535 -n 1)
  ssh beluga3 -L $LOCALPORT:$REMOTEHOST:$REMOTEPORT -N -f && break
done || { echo "Giving up forwarding license port after $i attempts..."; exit 1; }


# load modules and activate env
module load python/3.8
module load scipy-stack/2022a
source $HOME/mdi-optuna/bin/activate


OPTUNA_STUDY_NAME=utilismart_diffusion_user0_run1

#OPTUNA_DB=sqlite:///$HOME/${OPTUNA_STUDY_NAME}.db
OPTUNA_DB=mysql://optuna:Optuna#1234@$LOCALHOST:$LOCALPORT/OptunaDB

# Launch your script, giving it as arguments the database file and the study name
python hpo_utilismart.py --optuna-db $OPTUNA_DB --optuna-study-name $OPTUNA_STUDY_NAME
