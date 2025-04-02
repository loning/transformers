# -*- coding: utf-8 -*-
"""简化版的模型工具类，用于测试"""

class PreTrainedModel:
    """简化版的预训练模型基类"""
    
    def __init__(self, config):
        self.config = config
    
    def _init_weights(self, module):
        """初始化权重"""
        pass
    
    def post_init(self):
        """模型初始化后操作"""
        pass
    
    def get_extended_attention_mask(self, attention_mask, input_shape):
        """
        创建扩展的注意力掩码
        简化版实现，返回原始掩码
        """
        return attention_mask
    
    def __call__(self, *args, **kwargs):
        """调用forward方法"""
        return self.forward(*args, **kwargs) 