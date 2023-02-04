# Dataset Processing
Please separate the dataset into source set, labeled target set, and unlabeled target set.  

# training for BSCA
If you want to train the model, for example, please run:
> CUDA_VISIBLE_DEVICES=7 python main.py --method BSCA --net resnet34 --source WHU --target RSSCN7 --steps 2001 --thr 0.5 --sample_per_class 3 --save_check


