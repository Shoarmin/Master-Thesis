#!/bin/sh
#you can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

# The default partition is the 'general' partition
#SBATCH --partition=general

# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=medium

# The default run (wall-clock) time is 1 minute
#SBATCH --time=10:00:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# The default number of CPUs per task is 1, however CPUs are always allocated per 2, so for a single task you should use
#SBATCH --cpus-per-task=1

# The default memory per node is 1024 megabytes (1GB)
#SBATCH --mem=32GB

#SBATCH --gres=gpu:a40:1

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

# Use this simple command to check that your sbatch settings are working (it should show the GPU that you requested)
/usr/bin/nvidia-smi

#Job name
#SBATCH --job-name=start

#Output file
module use /opt/insy/modulefiles
module load cuda/11.1 cudnn/11.1-8.0.5.39 miniconda/3.9
module list

# Your job commands go below here

#echo "Sourcing Ablation venv"
conda activate /tudelft.net/staff-bulk/ewi/insy/CYS/shoarmin/env/attack/
echo -ne "Executing script "
echo $1
echo -ne "Running on node "
hostname
echo "Standard output:"

srun python federated.py --data=tinyimage --local_ep=5 --bs=256 --num_agents=10 --rounds=100 --client_lr=0.01 --num_corrupt=1 --poison_frac=0.2 --climg_attack=0 --pattern=sig --delta_val=60 --delta_attack=60
#srun python federated.py --data=tinyimage --local_ep=5 --bs=256 --num_agents=10 --rounds=100 --client_lr=0.01 --num_corrupt=1 --poison_frac=0.2 --climg_attack=0 --pattern=sig --delta_val=60 --delta_attack=50
#srun python federated.py --data=tinyimage --local_ep=5 --bs=256 --num_agents=10 --rounds=100 --client_lr=0.01 --num_corrupt=1 --poison_frac=0.2 --climg_attack=0 --pattern=sig --delta_val=60 --delta_attack=40
#srun python federated.py --data=tinyimage --local_ep=5 --bs=256 --num_agents=10 --rounds=100 --client_lr=0.01 --num_corrupt=1 --poison_frac=0.2 --climg_attack=0 --pattern=sig --delta_val=60 --delta_attack=20
#srun python federated.py --data=tinyimage --local_ep=5 --bs=256 --num_agents=10 --rounds=100 --client_lr=0.01 --num_corrupt=1 --poison_frac=0.2 --climg_attack=0 --pattern=sig --delta_val=0 --delta_attack=0

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
