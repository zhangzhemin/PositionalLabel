# Positional-Label
This repo is the implementation of "Positional Label for Self-Supervised Vision Transformer". This repo is modified from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).

# Usage

The training method refers to [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), we mainly modify ViT_B.py in the "models" folder and the "train_one_epoch" function in main.py.

For example, to train Positional Label with 4 GPU, run:

python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --batch-size 256

