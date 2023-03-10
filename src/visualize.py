from bertviz import model_view
from utils import utils
import torch

model_path = "../experiments/211116/Best_211116_fromScratch_Regression_GPU2_Layer1_2021-11-16_17-30-37_DdA/checkpoints/model_best.pth"
model = torch.load(model_path, map_location=lambda storage, loc: storage)

# input tensor (1, 50, 8)
X = torch.ones([1, 50, 8], dtype=torch.float32)
print(X)
print()

# input padding masks (1, 50) (all True)
padding_masks = torch.ones([1, 50], dtype=torch.bool)
print(padding_masks)
print()

# outputs = model(X, padding_masks)
# print(outputs)
# print()

attention = torch.ones([2, 1, 8, 50, 50], dtype=torch.float32)
model_view(attention, X, include_layers=list(range(2)), include_heads=list(range(8)))
