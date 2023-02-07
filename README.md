This is pytorch code of the paper "Semi-supervised Bidirectional Alignment for Remote Sensing Cross-domain Scene Classification" of ISPRS-2023. If it is helpful, please kindly cite it as: 
> @article{HUANG2023192,  
> title = {Semi-supervised bidirectional alignment for Remote Sensing cross-domain scene classification},  
> journal = {ISPRS Journal of Photogrammetry and Remote Sensing},  
> volume = {195},  
> pages = {192-203},  
> year = {2023},  
> issn = {0924-2716},  
> doi = {https://doi.org/10.1016/j.isprsjprs.2022.11.013},  
> url = {https://www.sciencedirect.com/science/article/pii/S0924271622003069},  
> author = {Wei Huang and Yilei Shi and Zhitong Xiong and Qi Wang and Xiao Xiang Zhu},  
> }  


# Dataset Processing
Please separate the dataset into source set, labeled target set, and unlabeled target set.  

# training for BSCA
If you want to train the model, for example, please run:
> CUDA_VISIBLE_DEVICES=7 python main.py --method BSCA --net resnet34 --source WHU --target RSSCN7 --steps 2001 --thr 0.5 --sample_per_class 3 --save_check


