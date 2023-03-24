
import torch
import torchvision

from torch import nn

def create_vit_b16(num_classes: int):
  """
  Creates a ViT model and return the model with its transforms
  """

  vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

  vit_b16_model = torchvision.models.vit_b_16(weights=vit_weights)

  vit_transforms = vit_weights.transforms()

  # freeze all the layers
  for param in vit_b16_model.parameters():
    param.requires_grad = False
  
  # changing the head
  vit_b16_model.heads = nn.Sequential(
      nn.Linear(in_features=768, out_features=num_classes)
  )

  return vit_b16_model, vit_transforms
