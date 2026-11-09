#!/bin/bash
#SBATCH -A m4958 -C cpu -N 1
#SBATCH -o /pscratch/sd/d/danieltm/quirk_mu60/logs/evalchunk-%x-%j.out
# sbatch -q regular|debug -t T -J NAME sbatch_eval_chunked.sh PRED_DIR DATASET A B OUT [CHUNK PAR J]
srun -N1 -n1 -c 256 --mem=0 bash /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/mu60_eval/eval_chunked.sh "$@"
