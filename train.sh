#!/bin/bash
#SBATCH -A cvit_mobility
#SBATCH -n 1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=40G
#SBATCH --time=72:00:00
#SBATCH --nodelist=gnode29
#SBATCH --mail-user=vkhindkar@gmail.com
#SBATCH --mail-type=ALL

#activat conda env
conda activate my_env_vk
echo "conda env activated"

module load cuda/9.0

echo "training.."

#python da_trainval_net.py --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path}  --max_epochs 12
#python3 trainval_net_MEAA.py --cuda --net vgg16 --dataset cityscape --dataset_t bdd100k --use_tfb
#python3 trainval_net_MEAA.py --cuda --net vgg16 --dataset cityscape --dataset_t bdd100k --lr 0.0001 --use_tfb

#python3 trainval_net_MEAA.py --cuda --net vgg16 --dataset cityscape --dataset_t bdd100k --lr 0.0001 --load_name /home2/vkhindkar/DA-OD-MEAA-PyTorch-main/models/vgg16/cityscape/globallocal_target_bdd100k_eta_0.1_gc1_False_gc2_False_gc3_False_gamma_5_session_1_epoch_5_step_9999.pth --start_epoch 6 --r True --checkepoch 5 --checkpoint 5 --lr_decay_step 10

#python3 trainval_net_MEAA.py --cuda --net vgg16 --dataset cityscape --dataset_t bdd100k --lr 0.0001 --load_name /home2/vkhindkar/DA-OD-MEAA-PyTorch-main/models/vgg16/cityscape/globallocal_target_bdd100k_eta_0.1_gc1_False_gc2_False_gc3_False_gamma_5_session_1_epoch_10_step_9999.pth --start_epoch 11 --r True --checkepoch 10 --checkpoint 10 --lr_decay_step 10

#python3 trainval_net_MEAA.py --cuda --net vgg16 --dataset cityscape --dataset_t bdd100k --lr 0.0001 --load_name /home2/vkhindkar/DA-OD-MEAA-PyTorch-main/models/vgg16/cityscape/globallocal_target_bdd100k_eta_0.1_gc1_False_gc2_False_gc3_False_gamma_5_session_1_epoch_13_step_9999.pth --start_epoch 14 --r True --checkepoch 13 --checkpoint 13 --lr_decay_step 10

#python3 trainval_net_MEAA.py --cuda --net vgg16 --dataset cityscape --dataset_t bdd100k --load_name /home2/vkhindkar/DA-OD-MEAA-PyTorch-main/models/vgg16/cityscape/globallocal_target_bdd100k_eta_0.1_gc1_False_gc2_False_gc3_False_gamma_5_session_1_epoch_2_step_9999.pth --start_epoch 3 --r True --checkepoch 2 --checkpoint 2 --lr_decay_step 2

#python3 trainval_net_MEAA.py --cuda --net vgg16 --dataset cityscape --dataset_t bdd100k --load_name /home2/vkhindkar/DA-OD-MEAA-PyTorch-main/models/vgg16/cityscape/globallocal_target_bdd100k_eta_0.1_gc1_False_gc2_False_gc3_False_gamma_5_session_1_epoch_3_step_9999.pth --start_epoch 4 --r True --checkepoch 3 --checkpoint 3 --lr_decay_step 2

#python3 trainval_net_MEAA.py --cuda --net vgg16 --dataset cityscape --dataset_t bdd100k --load_name /home2/vkhindkar/DA-OD-MEAA-PyTorch-main/models/vgg16/cityscape/globallocal_target_bdd100k_eta_0.1_gc1_False_gc2_False_gc3_False_gamma_5_session_1_epoch_5_step_9999.pth --start_epoch 6 --r True --checkepoch 5 --checkpoint 5 --lr_decay_step 2

#python3 trainval_net_MEAA.py --cuda --net vgg16 --dataset sim10k --dataset_t cityscape --use_tfb

#python3 trainval_net_MEAA.py --cuda --net vgg16 --dataset sim10k --dataset_t cityscape --load_name /home2/vkhindkar/DA-OD-MEAA-PyTorch-main/models/vgg16/sim10k/globallocal_target_cityscape_eta_0.1_gc1_False_gc2_False_gc3_False_gamma_5_session_1_epoch_2_step_9999.pth --start_epoch 3 --r True --checkepoch 2 --checkpoint 2

#python3 trainval_net_MEAA.py --cuda --net res101 --dataset pascal_voc --dataset_t clipart --use_tfb

#python3 trainval_net_MEAA.py --cuda --net res101 --dataset pascal_voc --dataset_t clipart --load_name /home2/vkhindkar/DA-OD-MEAA-PyTorch-main/models/res101/pascal_voc/globallocal_target_clipart_eta_0.1_gc1_False_gc2_False_gc3_False_gamma_5_session_1_epoch_12_step_9999.pth --start_epoch 13 --r True --checkepoch 12 --checkpoint 12 --use_tfb

#python3 trainval_net_MEAA.py --cuda --net vgg16 --dataset pascal_voc --dataset_t clipart --lr 0.0001 --use_tfb

#python3 trainval_net_MEAA.py --cuda --net vgg16 --dataset pascal_voc_cycleclipart --dataset_t clipart --use_tfb

#python3 trainval_net_MEAA.py --cuda --net vgg16 --dataset cityscape --dataset_t foggy_cityscape

#python3 trainval_net_MEAA.py --cuda --net res101 --dataset pascal_voc_0712 --dataset_t clipart

python3 trainval_net_MEAA.py --cuda --net res101 --dataset pascal_voc_0712 --dataset_t clipart --lr_decay_step 2

echo "Job completed"
~                                                                                                           
