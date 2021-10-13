#!/bin/bash
#SBATCH -A cvit_mobility
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=48:00:00
#SBATCH --nodelist=gnode10
#SBATCH --mail-user=vkhindkar@gmail.com
#SBATCH --mail-type=ALL

#activat conda env
conda activate my_env_vk
echo "conda env activated"

module load cuda/9.0

echo "training.."

#python da_trainval_net.py --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path}  --max_epochs 12
python3 trainval_net_MEAA.py --cuda --net vgg16 --dataset cityscape --dataset_t foggy_cityscape --use_tfb
