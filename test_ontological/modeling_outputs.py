# -*- coding: utf-8 -*-
"""简化版的模型输出类，用于测试"""

class ModelOutput:
    """简化版的ModelOutput基类"""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class BaseModelOutputWithPooling(ModelOutput):
    """包含池化输出的基本模型输出"""
    
    def __init__(
        self,
        last_hidden_state=None,
        pooler_output=None,
        hidden_states=None,
        attentions=None,
    ):
        super().__init__(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=hidden_states,
            attentions=attentions,
        )

class SequenceClassifierOutput(ModelOutput):
    """序列分类模型的输出"""
    
    def __init__(
        self,
        loss=None,
        logits=None,
        hidden_states=None,
        attentions=None,
    ):
        super().__init__(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        ) 