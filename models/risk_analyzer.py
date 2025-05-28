#!/usr/bin/env python3
"""
AI风险分析器 - 完全基于AI智能评估
=====================================

Copyright (c) 2025 superjackche
GitHub: https://github.com/superjackche/audio_ai
Licensed under the MIT License. See LICENSE file for details.
"""

import re
import json
import hashlib
from typing import Dict, List, Any
import logging
from datetime import datetime

class PoliticalRiskAnalyzer:
    """政治风险分析器 - 完全基于AI智能评分"""
    
    def __init__(self, model_manager=None):
        self.logger = logging.getLogger(__name__)
        self.model_manager = model_manager
        
        # AI评估的提示词模板
        self.ai_prompt_template = self._create_ai_prompt_template()
    
    def _create_ai_prompt_template(self):
        """创建AI评估的提示词模板"""
        return """你是一个专业的内容风险评估专家。请客观分析以下文本内容的政治风险等级，并给出0-100分的真实风险评分。

评分标准：
- 0-15分：无风险，内容安全正常，如：日常生活、学习工作、娱乐休闲等
- 16-30分：低风险，轻微敏感但可接受，如：一般时事讨论、历史介绍等
- 31-50分：中风险，存在一定争议性，如：政策分析、社会现象评论等
- 51-70分：中高风险，包含较明显的敏感内容，如：政治制度讨论、意识形态差异等
- 71-85分：高风险，存在明显政治敏感内容，如：政权批评、体制质疑等
- 86-100分：极高风险，包含严重政治风险内容，如：煽动性言论、极端思想等

请根据以下文本内容进行评估：

待分析文本：
{text}

请以JSON格式回复，必须包含以下字段：
{{
    "risk_score": 具体分数(0-100整数),
    "risk_level": "风险等级名称",
    "risk_factors": ["检测到的风险因素1", "风险因素2"],
    "key_concerns": ["主要关注点1", "关注点2"],
    "recommendations": ["建议1", "建议2"],
    "reasoning": "详细评估理由"
}}"""
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        分析文本的政治风险
        完全基于AI模型的智能评估，不使用关键词匹配
        
        Args:
            text: 需要分析的文本内容
            
        Returns:
            包含风险分析结果的字典
        """
        if not text or not text.strip():
            return self._create_empty_result("文本内容为空")
        
        try:
            # 优先使用AI模型进行评估
            if self.model_manager:
                ai_result = self._ai_assess_risk(text)
                if ai_result:
                    self.logger.info(f"AI评估完成，风险评分: {ai_result.get('risk_score', 'N/A')}")
                    return ai_result
            
            # 如果AI评估失败，使用备用的简化分析
            self.logger.warning("AI评估不可用，使用备用分析方法")
            return self._fallback_analysis(text)
            
        except Exception as e:
            self.logger.error(f"风险分析出错: {str(e)}")
            return self._create_error_result(str(e))
    
    def _ai_assess_risk(self, text: str) -> Dict[str, Any]:
        """使用AI模型评估风险"""
        try:
            # 构建AI评估提示
            prompt = self.ai_prompt_template.format(text=text)
            
            # 调用AI模型生成评估结果
            ai_response = self.model_manager.generate_response(prompt)
            
            if not ai_response:
                self.logger.warning("AI模型返回空响应")
                return None
            
            # 解析AI返回的JSON结果
            parsed_result = self._parse_ai_response(ai_response)
            
            if parsed_result:
                # 标准化结果格式
                return self._standardize_ai_result(parsed_result, text)
            else:
                self.logger.warning("AI响应解析失败")
                return None
                
        except Exception as e:
            self.logger.error(f"AI评估失败: {str(e)}")
            return None
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """解析AI返回的JSON响应"""
        try:
            # 尝试直接解析JSON
            if response.strip().startswith('{') and response.strip().endswith('}'):
                return json.loads(response)
            
            # 尝试从响应中提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            
            # 如果无法解析JSON，尝试从响应中提取关键信息
            return self._extract_info_from_text(response)
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON解析失败: {str(e)}")
            return self._extract_info_from_text(response)
        except Exception as e:
            self.logger.error(f"响应解析错误: {str(e)}")
            return None
    
    def _extract_info_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取风险评估信息"""
        try:
            result = {}
            
            # 提取风险评分
            score_match = re.search(r'(?:风险评分|risk_score|评分)[：:]\s*(\d+)', text, re.IGNORECASE)
            if score_match:
                result['risk_score'] = int(score_match.group(1))
            else:
                # 尝试查找数字模式
                numbers = re.findall(r'\b(\d+)分\b', text)
                if numbers:
                    result['risk_score'] = int(numbers[0])
                else:
                    result['risk_score'] = 25  # 默认低风险评分
            
            # 提取风险等级
            level_patterns = [
                r'(?:风险等级|risk_level)[：:]\s*"?([^"，,\n]+)"?',
                r'(无风险|低风险|中风险|中高风险|高风险|极高风险)',
                r'(安全|较安全|一般|较高风险|危险|极度危险)'
            ]
            
            for pattern in level_patterns:
                level_match = re.search(pattern, text, re.IGNORECASE)
                if level_match:
                    result['risk_level'] = level_match.group(1).strip().strip('"')
                    break
            
            if 'risk_level' not in result:
                result['risk_level'] = self._score_to_level(result.get('risk_score', 25))
            
            # 提取其他信息
            result['risk_factors'] = re.findall(r'风险因素[：:].*?([^，,。.\n]+)', text)[:3]
            result['key_concerns'] = re.findall(r'关注点[：:].*?([^，,。.\n]+)', text)[:3]
            result['recommendations'] = ["建议进一步人工审核", "根据具体情况调整内容"]
            result['reasoning'] = "基于AI文本分析得出的评估结果"
            
            return result
            
        except Exception as e:
            self.logger.error(f"信息提取失败: {str(e)}")
            return None
    
    def _standardize_ai_result(self, ai_result: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """标准化AI评估结果"""
        try:
            # 确保风险评分在合理范围内
            risk_score = ai_result.get('risk_score', 25)
            if not isinstance(risk_score, int) or risk_score < 0 or risk_score > 100:
                risk_score = max(0, min(100, int(risk_score) if str(risk_score).isdigit() else 25))
            
            # 根据评分确定风险等级
            risk_level = ai_result.get('risk_level', self._score_to_level(risk_score))
            
            # 确保列表字段存在且为列表
            risk_factors = ai_result.get('risk_factors', [])
            if not isinstance(risk_factors, list):
                risk_factors = [str(risk_factors)] if risk_factors else []
            
            key_concerns = ai_result.get('key_concerns', [])
            if not isinstance(key_concerns, list):
                key_concerns = [str(key_concerns)] if key_concerns else []
            
            recommendations = ai_result.get('recommendations', [])
            if not isinstance(recommendations, list):
                recommendations = [str(recommendations)] if recommendations else ["建议进一步审核"]
            
            # 构建标准化结果
            standardized_result = {
                'total_score': risk_score, # 确保此键存在
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_factors': risk_factors[:5],  # 最多5个风险因素
                'key_concerns': key_concerns[:5],  # 最多5个关注点
                'recommendations': recommendations[:3],  # 最多3个建议
                'reasoning': ai_result.get('reasoning', '基于AI智能分析'),
                'confidence': self._calculate_confidence(ai_result),
                'analysis_method': 'AI智能评估',
                'timestamp': datetime.now().isoformat(),
                'text_hash': hashlib.md5(original_text.encode()).hexdigest()[:16]
            }
            
            return standardized_result
            
        except Exception as e:
            self.logger.error(f"结果标准化失败: {str(e)}")
            return self._create_error_result(f"结果标准化失败: {str(e)}")
    
    def _calculate_confidence(self, ai_result: Dict[str, Any]) -> float:
        """计算评估结果的置信度"""
        try:
            confidence = 0.8  # 基础置信度
            
            # 如果有详细的风险因素，增加置信度
            if ai_result.get('risk_factors') and len(ai_result['risk_factors']) > 0:
                confidence += 0.1
            
            # 如果有具体的关注点，增加置信度
            if ai_result.get('key_concerns') and len(ai_result['key_concerns']) > 0:
                confidence += 0.05
            
            # 如果有详细的推理说明，增加置信度
            reasoning = ai_result.get('reasoning', '')
            if reasoning and len(reasoning) > 20:
                confidence += 0.05
            
            return min(confidence, 1.0)
            
        except Exception:
            return 0.7  # 默认置信度
    
    def _score_to_level(self, score: int) -> str:
        """将数字评分转换为风险等级"""
        if score <= 15:
            return "无风险"
        elif score <= 30:
            return "低风险"
        elif score <= 50:
            return "中风险"
        elif score <= 70:
            return "中高风险"
        elif score <= 85:
            return "高风险"
        else:
            return "极高风险"
    
    def _fallback_analysis(self, text: str) -> Dict[str, Any]:
        """备用的简化分析方法（当AI不可用时使用）"""
        try:
            # 基于文本长度和内容复杂度的简单评估
            text_length = len(text)
            
            # 基础风险评分（保守估计）
            base_score = 20  # 默认低风险
            
            # 根据文本特征调整评分
            if text_length > 500:
                base_score += 5  # 长文本可能包含更多信息
            
            # 检查是否包含一些通用的敏感词汇指示符
            sensitive_indicators = ['政治', '政府', '制度', '政策', '领导', '国家', '社会', '改革']
            found_indicators = sum(1 for indicator in sensitive_indicators if indicator in text)
            
            if found_indicators > 0:
                base_score += found_indicators * 3  # 每个指示符增加3分
            
            # 确保评分在合理范围内
            final_score = max(10, min(50, base_score))  # 备用方法最高给50分（中风险）
            
            return {
                'total_score': final_score, # 确保此键存在
                'risk_score': final_score,
                'risk_level': self._score_to_level(final_score),
                'risk_factors': ['使用备用分析方法'],
                'key_concerns': ['建议使用AI评估获得更准确结果'],
                'recommendations': ['尽快恢复AI评估功能', '进行人工复核'],
                'reasoning': '使用备用简化分析方法，建议使用AI获得更准确评估',
                'confidence': 0.4,  # 备用方法置信度较低
                'analysis_method': '备用简化评估',
                'timestamp': datetime.now().isoformat(),
                'text_hash': hashlib.md5(text.encode()).hexdigest()[:16]
            }
            
        except Exception as e:
            self.logger.error(f"备用分析失败: {str(e)}")
            return self._create_error_result(f"备用分析失败: {str(e)}")
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """创建空文本的结果"""
        return {
            'risk_score': 0,
            'risk_level': '无风险',
            'risk_factors': [],
            'key_concerns': [],
            'recommendations': [],
            'reasoning': reason,
            'confidence': 1.0,
            'analysis_method': '空文本检测',
            'timestamp': datetime.now().isoformat(),
            'text_hash': ''
        }
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """创建错误情况的结果"""
        return {
            'risk_score': 30,  # 出错时给一个保守的中低风险评分
            'risk_level': '中风险',
            'risk_factors': ['分析过程出现错误'],
            'key_concerns': ['无法完成准确评估'],
            'recommendations': ['需要人工审核', '检查系统状态'],
            'reasoning': f'分析出错: {error_msg}',
            'confidence': 0.1,
            'analysis_method': '错误处理',
            'timestamp': datetime.now().isoformat(),
            'text_hash': '',
            'error': error_msg
        }
    
    def get_risk_statistics(self) -> Dict[str, Any]:
        """获取风险分析统计信息"""
        return {
            'analyzer_type': 'AI智能风险评估器',
            'version': '2.0',
            'ai_enabled': self.model_manager is not None,
            'evaluation_method': 'AI智能评分 + 备用简化分析',
            'supported_features': [
                '智能风险评分 (0-100)',
                '风险等级分类',
                '风险因素识别',
                '关注点提取',
                '改进建议',
                '置信度评估'
            ]
        }
