# PositionalLabel
This repo is the official implementation of ["Positional Label for Self-Supervised Vision Transformer"](https://arxiv.org/pdf/2206.04981.pdf). Modified from [Swin-ViT](https://github.com/microsoft/Swin-Transformer). We mainly modified ViT_B.py and swin_transformer.py in the models folder.

# Usage
To train the Positional Label on ImageNet from scratch, run:

python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --batch-size 320

or

python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --batch-size 256
