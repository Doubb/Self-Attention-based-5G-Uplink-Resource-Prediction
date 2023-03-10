import os
import sys
import time
import pickle
import json

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch._C import MobileOptimizerType

# Project modules
from options import Options
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from utils import utils
from datasets.data import data_factory, Normalizer
from datasets.datasplit import split_dataset
from models.ts_transformer import model_factory
from models.loss import get_loss_module
from optimizers import get_optimizer

model_path = '../220129_Layer3_LN_RAdam_relu.pt'
model = torch.load(model_path)
model.to('cpu')
model.eval()

quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
#torch.save(quantized_model, 'transformer_model_quantized.pt')

torchscript_model = torch.jit.script(quantized_model)
torchscript_model_optimized = optimize_for_mobile(torchscript_model)
torch.jit.save(torchscript_model_optimized, 'transformer_model_optimized_total_power.pt')
