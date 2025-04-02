# -*- coding: utf-8 -*-
"""基于宇宙本论（仅使用XOR、SHIFT和FLIP操作）的PyTorch Transformer模型简化版。"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入我们的简化版模块
from modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput
from modeling_utils import PreTrainedModel
from utils import logging
from simplified_config import OntologicalTransformerConfig 