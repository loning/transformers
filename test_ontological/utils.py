# -*- coding: utf-8 -*-
"""简化版的工具函数，用于测试"""

class logging:
    """简化版的日志类"""
    
    @staticmethod
    def get_logger(name):
        """返回一个简单的logger"""
        class SimpleLogger:
            def warning_once(self, msg):
                pass
        return SimpleLogger() 