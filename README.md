# PositionalLabel
This repo is the official implementation of ["Positional Label for Self-Supervised Vision Transformer"](https://arxiv.org/pdf/2206.04981.pdf). Modified from [Swin-ViT](https://github.com/microsoft/Swin-Transformer). We mainly modified ViT_B.py and swin_transformer.py in the models folder.

## Introduction

**Positional Label** is initially described in [arxiv]( https://arxiv.org/pdf/2206.04981.pdf). In our work we propose to train ViT to recognize the positional label of patches of the input image, this apparently simple task actually yields a meaningful self-supervisory task. Based on previous work on ViT positional encoding, we propose two positional labels dedicated to 2D images including absolute position and relative position. Our positional labels can be easily plugged into various current ViT variants. It can work in two ways: (a) As an auxiliary training target for vanilla ViT for better performance. (b) Combine the self-supervised ViT to provide a more powerful self-supervised signal for semantic feature learning.

![APL](figures/ APL.jpg)


# Usage
To train the Positional Label on ImageNet from scratch, run:

python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --batch-size 320

or

python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --batch-size 256
