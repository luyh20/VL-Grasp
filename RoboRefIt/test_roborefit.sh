export CUDA_VISIBLE_DEVICES=4,5,6,7
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main_vg.py --eval True --auto_resume --aux_loss --data_root data/final_dataset --test_split testA --backbone resnet50  --epochs 90 --batch_size 16 --dataset roborefit --lr 0.00001 --masks True --img_size 640 --img_type 'RGB' --output_dir outputs/roborefit
