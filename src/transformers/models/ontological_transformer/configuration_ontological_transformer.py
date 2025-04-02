# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""基于宇宙本论的Transformer模型配置"""

from collections import OrderedDict
from typing import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class OntologicalTransformerConfig(PretrainedConfig):
    r"""
    这是用于存储`OntologicalTransformerModel`的配置类。它用于实例化符合宇宙本论的Transformer模型，
    该模型仅使用XOR、SHIFT和FLIP操作来进行计算。
    
    配置对象继承自`PretrainedConfig`，可用于控制模型输出。有关更多信息，请阅读`PretrainedConfig`的文档。

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            词汇表大小。定义调用`OntologicalTransformerModel`时`inputs_ids`可以表示的不同标记数量。
        hidden_size (`int`, *optional*, defaults to 768):
            编码器层和池化层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer编码器中的隐藏层数。
        num_attention_heads (`int`, *optional*, defaults to 12):
            Transformer编码器中每个注意力层的注意力头数。
        intermediate_size (`int`, *optional*, defaults to 3072):
            Transformer编码器中"中间"（通常称为前馈）层的维度。
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。如果是字符串，支持"gelu"、"relu"和"selu"。
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            嵌入、编码器和池化器中所有全连接层的丢弃概率。
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            注意力概率的丢弃率。
        max_position_embeddings (`int`, *optional*, defaults to 512):
            此模型可能使用的最大序列长度。通常设置一个足够大的值以防万一（例如512、1024或2048）。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态分布的标准差。
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            层归一化层使用的epsilon。
        use_xor_attention (`bool`, *optional*, defaults to True):
            是否在注意力计算中使用XOR操作。
        use_shift_ffn (`bool`, *optional*, defaults to True):
            是否在前馈网络中使用SHIFT操作。
        use_flip_output (`bool`, *optional*, defaults to True):
            是否在输出层中使用FLIP操作。
        pad_token_id (`int`, *optional*, defaults to 0):
            填充标记的ID。
        is_decoder (`bool`, *optional*, defaults to `False`):
            模型是否用作解码器。如果为`False`，则模型用作编码器。
        use_cache (`bool`, *optional*, defaults to `True`):
            模型是否应返回最后的键/值注意（并非所有模型都使用）。仅在`config.is_decoder=True`时相关。
        classifier_dropout (`float`, *optional*):
            分类头的丢弃率。

    示例:

    ```python
    >>> from transformers import OntologicalTransformerConfig, OntologicalTransformerModel

    >>> # 初始化默认风格的配置
    >>> configuration = OntologicalTransformerConfig()

    >>> # 从配置初始化模型（随机权重）
    >>> model = OntologicalTransformerModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    model_type = "ontological_transformer"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_xor_attention=True,
        use_shift_ffn=True,
        use_flip_output=True,
        pad_token_id=0,
        is_decoder=False,
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_xor_attention = use_xor_attention
        self.use_shift_ffn = use_shift_ffn
        self.use_flip_output = use_flip_output
        self.is_decoder = is_decoder
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


class OntologicalTransformerOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )


__all__ = ["OntologicalTransformerConfig", "OntologicalTransformerOnnxConfig"] 