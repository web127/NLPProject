"""Check what's in the checkpoint"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import torch

ckpt_path = ROOT / "homework_outputs/checkpoints/best_linear.pt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print("=" * 70)
print("Checkpoint keys:", list(ckpt.keys()))
print("=" * 70)
print(f"Epoch: {ckpt['epoch']}")
print(f"Use CRF: {ckpt['use_crf']}")
print(f"Val F1: {ckpt['val_entity_f1']}")
print(f"\nArgs: {ckpt['args']}")
print(f"\nLabel2id: {ckpt['label2id']}")
print(f"\nId2label: {ckpt['id2label']}")

# 看state_dict
print("\n" + "=" * 70)
print("Model state dict keys:")
for key in list(ckpt['state_dict'].keys())[:5]:
    print(f"  {key}")

print("\n" + "=" * 70)
print("Checking classifier shape")
classifier_key = 'classifier.weight'
if classifier_key in ckpt['state_dict']:
    print(f"Classifier weight shape: {ckpt['state_dict'][classifier_key].shape}")
