# Double-DIP: Deep Image Priors

A partial implementation of the paper ["Double-DIP": Unsupervised Image Decomposition via Coupled Deep-Image-Priors](http://openaccess.thecvf.com/content_CVPR_2019/html/Gandelsman_Double-DIP_Unsupervised_Image_Decomposition_via_Coupled_Deep-Image-Priors_CVPR_2019_paper.html)

## Requirements

Tests were run on a 1080-Ti GPU
- pytorch = 1.2.0
- torchvision == 0.3.0a0+9168476
- python 3.6

## Model

The model code is in `dipmodel.py`. It is based on the description provided in the original [Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior) paper and supplementary material.

## Experiments

I attempted to replicate two sets of experiments from the paper. The first is the experiments using a single DIP model to learn blended textures and natural images, as described in Section 2.1 of the paper. Those experiments are in the notebook `blend-testing.ipynb`. I also tried to reproduce the segmentation results from the paper. That experiment is in `segmentation.ipynb`.

## Status

The code in the notebooks runs and prodcues results, but the results don't match the paper. The blending results contradict the paper, as most of the blended images had lower error than the single images they were composed of. I'm unsure what causes this descrepency. I was also unable to get good segmentation results. It seems to be tricky to get it to work right.
