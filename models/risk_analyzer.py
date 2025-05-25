import re
import jieba
import jieba.analyse
from typing import Dict, List, Any, Tuple
from config.settings import RISK_KEYWORDS, SCORING_WEIGHTS
import logging
from datetime import datetime

class PoliticalRiskAnalyzer:
    """政治风险分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 初始化jieba分词
        jieba.initialize()
        
        # 加载自定义词典（政治敏感词）
        self._load_custom_dict()
    
    def _load_custom_dict(self):
        """加载自定义政治词典"""
        try:
            # 将所有风险关键词添加到jieba词典
            for risk_level, keywords in RISK_KEYWORDS.items():
                for keyword in keywords:
                    jieba.add_word(keyword, freq=1000)
            self.logger.info("自定义政治词典加载完成")
        except Exception as e:
            self.logger.error(f"加载自定义词典失败: {e}")
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """综合分析文本的政治风险"""
        # 基础信息
        analysis_time = datetime.now().isoformat()
        
        # 各项分析
        keyword_analysis = self._analyze_keywords(text)
        context_analysis = self._analyze_context(text)
        frequency_analysis = self._analyze_frequency(text)
        sentiment_analysis = self._analyze_sentiment(text)
        
        # 计算总分
        total_score = self._calculate_total_score(
            keyword_analysis["score"],
            context_analysis["score"],
            frequency_analysis["score"],
            sentiment_analysis["score"]
        )
        
        # 确定风险等级
        risk_level = self._determine_risk_level(total_score)
        
        return {
            "analysis_time": analysis_time,
            "text_length": len(text),
            "total_score": total_score,
            "risk_level": risk_level,
            "keyword_analysis": keyword_analysis,
            "context_analysis": context_analysis,
            "frequency_analysis": frequency_analysis,
            "sentiment_analysis": sentiment_analysis,
            "recommendations": self._generate_recommendations(total_score, risk_level)
        }
    
    def _analyze_keywords(self, text: str) -> Dict[str, Any]:
        """关键词分析"""
        found_keywords = {}
        total_score = 0
        
        for risk_level, keywords in RISK_KEYWORDS.items():
            level_score = {"高风险": 80, "中风险": 50, "低风险": 20}[risk_level]
            found_in_level = []
            
            for keyword in keywords:
                count = text.count(keyword)
                if count > 0:
                    found_in_level.append({
                        "keyword": keyword,
                        "count": count,
                        "positions": [m.start() for m in re.finditer(keyword, text)]
                    })
                    total_score += level_score * count
        
            if found_in_level:
                found_keywords[risk_level] = found_in_level
        
        return {
            "score": min(total_score, 100),  # 限制最高分100
            "found_keywords": found_keywords,
            "total_keyword_matches": sum(
                len(keywords) for keywords in found_keywords.values()
            )
        }
    
    def _analyze_context(self, text: str) -> Dict[str, Any]:
        """上下文语义分析"""
        # 使用jieba提取关键词
        keywords = jieba.analyse.extract_tags(text, topK=20, withWeight=True)
        
        # 政治相关词汇模式
        political_patterns = [
            r'政[治府策权]',
            r'制度.*问题',
            r'民主.*[化制]',
            r'[独分裂]立.*[主义运动]',
            r'意识形态.*[问题冲突]',
            r'价值观.*[差异冲突]'
        ]
        
        pattern_matches = []
        context_score = 0
        
        for pattern in political_patterns:
            matches = re.findall(pattern, text)
            if matches:
                pattern_matches.extend(matches)
                context_score += len(matches) * 15
        
        return {
            "score": min(context_score, 100),
            "key_phrases": [kw[0] for kw in keywords[:10]],
            "political_patterns": pattern_matches,
            "semantic_density": len(pattern_matches) / max(len(text.split()), 1)
        }
    
    def _analyze_frequency(self, text: str) -> Dict[str, Any]:
        """频率分析"""
        words = jieba.lcut(text)
        word_freq = {}
        
        # 统计政治相关词汇频率
        political_words = []
        for risk_level, keywords in RISK_KEYWORDS.items():
            political_words.extend(keywords)
        
        for word in words:
            if word in political_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 计算频率得分
        total_political_words = sum(word_freq.values())
        frequency_ratio = total_political_words / max(len(words), 1)
        frequency_score = min(frequency_ratio * 1000, 100)  # 标准化到100分
        
        return {
            "score": frequency_score,
            "political_word_count": total_political_words,
            "total_word_count": len(words),
            "frequency_ratio": frequency_ratio,
            "high_frequency_words": sorted(
                word_freq.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """情感倾向分析"""
        # 简单的情感词典
        negative_words = [
            "批评", "反对", "质疑", "抨击", "谴责", "否定",
            "错误", "失败", "问题", "危机", "冲突", "矛盾"
        ]
        
        positive_words = [
            "支持", "赞成", "肯定", "成功", "进步", "发展",
            "和谐", "稳定", "团结", "合作", "友好", "积极"
        ]
        
        negative_count = sum(text.count(word) for word in negative_words)
        positive_count = sum(text.count(word) for word in positive_words)
        
        # 计算情感倾向分数（负面情感越多风险越高）
        sentiment_score = 0
        if negative_count > positive_count:
            sentiment_score = min((negative_count - positive_count) * 10, 100)
        
        return {
            "score": sentiment_score,
            "negative_count": negative_count,
            "positive_count": positive_count,
            "sentiment_tendency": "negative" if negative_count > positive_count else "positive"
        }
    
    def _calculate_total_score(self, keyword_score: float, context_score: float, 
                              frequency_score: float, sentiment_score: float) -> float:
        """计算总体风险分数"""
        total_score = (
            keyword_score * SCORING_WEIGHTS["keyword_score"] +
            context_score * SCORING_WEIGHTS["context_score"] +
            frequency_score * SCORING_WEIGHTS["frequency_score"] +
            sentiment_score * SCORING_WEIGHTS["sentiment_score"]
        )
        return round(total_score, 2)
    
    def _determine_risk_level(self, score: float) -> str:
        """根据分数确定风险等级"""
        if score >= 70:
            return "高风险"
        elif score >= 40:
            return "中风险"
        elif score >= 20:
            return "低风险"
        else:
            return "无风险"
    
    def _generate_recommendations(self, score: float, risk_level: str) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if risk_level == "高风险":
            recommendations.extend([
                "建议立即暂停相关内容讲授",
                "需要进行内容审查和修正",
                "建议咨询相关专业人员",
                "加强意识形态培训"
            ])
        elif risk_level == "中风险":
            recommendations.extend([
                "建议调整表达方式",
                "注意中性化描述",
                "避免价值判断性语言",
                "增加客观分析视角"
            ])
        elif risk_level == "低风险":
            recommendations.extend([
                "总体表达较为安全",
                "可适当优化措辞",
                "保持客观中立立场"
            ])
        else:
            recommendations.append("内容表达规范，无明显风险")
        
        return recommendations
