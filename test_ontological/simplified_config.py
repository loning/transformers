# -*- coding: utf-8 -*-
"""简化版的配置文件，用于测试"""

class PretrainedConfig:
    """简化版的PretrainedConfig类"""
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def __repr__(self):
        return str(self.__dict__)


class OntologicalTransformerConfig(PretrainedConfig):
    """
    基于宇宙本论的Transformer模型配置
    """
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
        output_attentions=False,
        output_hidden_states=False,
        classifier_dropout=None,
        num_labels=2,
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
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.classifier_dropout = classifier_dropout
        self.num_labels = num_labels 