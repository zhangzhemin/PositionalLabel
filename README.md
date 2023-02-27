# PositionalLabel
This repo is the implementation of "Positional Label for Self-Supervised Vision Transformer". Modified from [Swin-ViT](https://github.com/microsoft/Swin-Transformer). We mainly modified ViT_B.py and swin_transformer.py in the models folder.

# Usage
To train the Positional Label on ImageNet from scratch, run:

python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --batch-size 256
