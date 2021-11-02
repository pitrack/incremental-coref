"""
Usage:  python clear_optims.py ckpt_1 [ckpt_2, ...]

Iterates through a list of checkpoints and deletes their optimizer states,
effectively reducing the size of the checkpoint to save disk space
"""
import torch
import sys


def clear(log_path):
  try:
    checkpoint = torch.load(log_path, map_location="cpu")
    print(f"Found checkpoint at {log_path}, loading instead.")
  except:
    print(f"Checkpoint not found at {log_path}")
    return
  if "encoder_optimizer" in checkpoint:
    del checkpoint["encoder_optimizer"]
  if "optimizer" in checkpoint:
    del checkpoint["optimizer"]
  torch.save(checkpoint, log_path)

for ckpt in sys.argv[1:]:
  clear(ckpt)
