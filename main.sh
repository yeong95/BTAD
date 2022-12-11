#!/bin/bash

# CUDA_VISIBLE_DEVICES=1 python train.py --phase train --dataset_path ../mvtec --category bottle --project_root_path ./ --coreset_sampling_ratio 0.01

# Mine
# sever 2
# python train_yk.py --phase train e--dataset_path /tf/Battery_data/무지부_코팅부_테이프제외2/ --category '코팅부' --project_root_path ./ --feature_level image

# # server 1 patchcore 
# CUDA_VISIBLE_DEVICES=0 python train_yk.py --phase test --dataset_path ../무지부_코팅부_테이프제외2/ --category '무지부_코팅부' --project_root_path ./ --feature_level patch \
# --n_neighbors 9 --distance L2 --train_folder train_downsampling3 --test_folder test --coreset_sampling_ratio 0.01

# server 1 patchcore - 무지부_코팅부
# CUDA_VISIBLE_DEVICES=1 python train_yk.py --phase test --dataset_path ../무지부_코팅부_테이프제외2/ --category '무지부_코팅부' --project_root_path ./ --feature_level image \
# --n_neighbors 10 --distance cosine --train_folder train_with_aug --test_folder test 

# oce image embedding - 무지부 
# CUDA_VISIBLE_DEVICES=1 python train_with_oce.py --phase train --dataset_path ../무지부_코팅부_테이프제외2/ --category '무지부' --project_root_path ./ --feature_level image \
# --n_neighbors 10 --distance L2 --train_folder train_cluster_10%_10times --test_folder valid_test_3 --network res50

# oce image embedding - 코팅부 
CUDA_VISIBLE_DEVICES=1 python train_with_oce.py --phase train --dataset_path ../무지부_코팅부_테이프제외2/ --category '코팅부' --project_root_path ./ --feature_level image \
--n_neighbors 10 --distance L2 --train_folder train_cluster_10%_10times --test_folder valid_test_2 --network res34

# oce image embedding - 무지부_코팅부
# CUDA_VISIBLE_DEVICES=1 python train_with_oce.py --phase test --dataset_path ../무지부_코팅부_테이프제외2/ --category '무지부_코팅부' --project_root_path ./ --feature_level image \
# --n_neighbors 10 --distance cosine --train_folder train_with_aug --test_folder test

# oce image embedding - cifar10
# CUDA_VISIBLE_DEVICES=0 python train_with_oce_benchmark.py --phase train --category 'cifar10' --project_root_path ./ --feature_level image \
# --n_neighbors 10 --distance L2 --normal_class 4 --network res18 --input_size 64

# # pretrained encoder - cifar10
# CUDA_VISIBLE_DEVICES=0 python train_benchmark.py --phase train --category 'cifar10' --project_root_path ./ --feature_level image \
# --n_neighbors 10 --distance cosine --normal_class 1



# oce embedding mvtec 
# CUDA_VISIBLE_DEVICES=1 python train_with_oce_mvtec.py --phase train --dataset_path ./MVTec --category toothbrush --project_root_path ./ --feature_level image \
# --n_neighbors 9 --distance cosine

# decoder patch mvtec
# CUDA_VISIBLE_DEVICES=1 python train_with_oce_decoder.py --phase train --dataset_path ./MVTec --category toothbrush --project_root_path ./ --feature_level patch \
# --n_neighbors 9 --distance L2 --coreset_sampling_ratio 0.01

# proposed method localization with mvtec 
# CUDA_VISIBLE_DEVICES=1 python train_with_oce_localization_mvtec.py --phase train --dataset_path ../mvtec --category toothbrush --project_root_path ./ \
# --n_neighbors 9 --distance cosine 
# list="carpet bottle hazelnut leather cable capsule grid pill transistor metal_nut screw zipper tile wood"
# for var in $list
# do  
#     echo $var
#     CUDA_VISIBLE_DEVICES=1 python train_with_oce_localization_mvtec.py --phase train --dataset_path ../mvtec --category $var --project_root_path ./ \
# --n_neighbors 9 --distance cosine         
# done  

# bottleneck patch 
# CUDA_VISIBLE_DEVICES=1 python train_with_oce_bottleneck_patch.py --phase train --dataset_path ../무지부_코팅부_테이프제외2/ --category '코팅부' \
# --project_root_path ./ --feature_level patch --n_neighbors 9 --distance L2 --train_folder train_cluster_10%_10times --test_folder valid_test_2 \
# --coreset_sampling_ratio 0.1

# for var in {2..10}
# do
#     # echo $var
#     python train_yk.py --phase test --dataset_path ../무지부_코팅부_테이프제외2/ --category '코팅부' --project_root_path ./ --feature_level image --n_neighbors $var --distance L2
# done