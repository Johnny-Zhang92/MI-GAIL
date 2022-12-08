#!/usr/bin/env bash

env_name_list=(Hopper-v2 HalfCheetah-v2 Ant-v2 Walker2d-v2)
#path_list=(4_value_dice_dac 3_value_dice_dac 2_value_dice_dac 1_value_dice_dac)
max_timesteps_list=(200000 200000 200000 200000)
seed_list=(0 2 4 6)
# change1: train_MI_GAIL
# change2: env_index
# change3: num_expert_trajectories
# change4: mi_weight
# change5: seed
# change6: GPU
number=0


mi_weight_list=(0.005 0.01 0.02 0.04)
num_expert_trajectories_list=(5 40)

for((i=0; i<=3; i++));
do
  for((j=0; j<=1; j++));
  do
    train_MI_GAIL=0
    train_DAC=0
    train_MI_GAIL_no_expert=1
    env_index=2
    num_expert_trajectories=${num_expert_trajectories_list[${j}]}
    mi_weight=${mi_weight_list[${i}]}
    seed_index=0
    gpu_num=0

    #cd ./${path_list[${env_index}]}
    #  pwd
    #python -m main_1 \
    nohup python -m main_1                            \
          --env_name "${env_name_list[${env_index}]}" \
          --GPU_num ${gpu_num}                        \
          --max_timesteps ${max_timesteps_list[${env_index}]}  \
          --seed ${seed_list[${seed_index}]}                              \
          --num_expert_trajectories ${num_expert_trajectories} \
          --train_MI_GAIL ${train_MI_GAIL}            \
          --train_DAC ${train_DAC}                     \
          --train_MI_GAIL_no_expert ${train_MI_GAIL_no_expert}               \
          --mi_weight ${mi_weight}                    \
          >> log/training_${env_name_list[${env_index}]}_[${seed_list[${seed_index}]}]_mi_weight_[${mi_weight}]_num_traj_[${num_expert_trajectories}]_MI_GAIL_[${train_MI_GAIL}].log 2>&1 &


    # return ID
    str="python -m main_1 --env_name ${env_name_list[${env_index}]} --GPU_num ${gpu_num} --max_timesteps ${max_timesteps_list[${env_index}]} --seed ${seed_list[${seed_index}]} --num_expert_trajectories ${num_expert_trajectories} --train_MI_GAIL ${train_MI_GAIL} --train_DAC ${train_DAC} --train_MI_GAIL_no_expert ${train_MI_GAIL_no_expert} --mi_weight ${mi_weight}"
    Program_ID=`ps aux | grep "${str}" | grep -v grep | awk '{print $2}'`
    #echo ${Program_ID}

    # Print current number
    let number++
    echo ${number} ", mi_weight:" ${mi_weight_list[${i}]}               \
    ",num_expert_trajectories:" ${num_expert_trajectories_list[${j}]}   \
     "Program_ID:" ${Program_ID};
  done
done


