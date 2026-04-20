"""
难度自适应检索模块 (Adaptive Retrieval)
========================================
功能：根据查询的难度动态调整检索策略

原理：
- 简单查询：减少检索数量，快速返回结果
- 困难查询：增加检索数量，提供更多上下文
- 不确定查询：可能触发多轮检索

优势：
- 效率提升：简单查询快速处理
- 准确率提升：困难查询获得更多支持
- 通用性：不依赖任务类型
"""

import logging
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass


@dataclass
class RetrievalStrategy:
    """
    检索策略配置

    定义不同难度级别对应的检索参数
    """
    k: int  # 检索数量
    use_rerank: bool  # 是否使用重排序
    confidence_threshold: float  # 置信度阈值


class DifficultyEstimator:
    """
    查询难度评估器

    根据多个指标评估查询的难度
    """

    def __init__(
        self,
        use_semantic_uncertainty: bool = True,
        use_length_feature: bool = True,
        use_specificity: bool = True
    ):
        """
        初始化难度评估器

        Args:
            use_semantic_uncertainty (bool): 是否使用语义不确定性特征
            use_length_feature (bool): 是否使用文本长度特征
            use_specificity (bool): 是否使用查询特异性特征
        """
        self.use_semantic_uncertainty = use_semantic_uncertainty
        self.use_length_feature = use_length_feature
        self.use_specificity = use_specificity

        logging.info("DifficultyEstimator initialized")

    def estimate(
        self,
        query: str,
        top_similarities: Optional[List[float]] = None,
        vocabulary_size: Optional[int] = None
    ) -> float:
        """
        评估查询难度

        Args:
            query (str): 查询文本
            top_similarities (List[float], optional): 检索结果的相似度分数列表
                用于计算语义不确定性
            vocabulary_size (int, optional): 语料库的词汇量
                用于计算查询特异性

        Returns:
            float: 难度分数，范围[0, 1]，越大表示越困难
                - 0.0 - 0.3: 简单查询
                - 0.3 - 0.7: 中等查询
                - 0.7 - 1.0: 困难查询

        原理：
            难度 = α * 语义不确定性 + β * 长度复杂度 + γ * 查询特异性
        """
        difficulty_scores = []
        weights = []

        # 特征1：语义不确定性
        if self.use_semantic_uncertainty and top_similarities is not None:
            semantic_uncertainty = self._calculate_semantic_uncertainty(
                top_similarities
            )
            difficulty_scores.append(semantic_uncertainty)
            weights.append(0.5)  # 语义不确定性权重最高

        # 特征2：查询长度复杂度
        if self.use_length_feature:
            length_complexity = self._calculate_length_complexity(query)
            difficulty_scores.append(length_complexity)
            weights.append(0.3)

        # 特征3：查询特异性
        if self.use_specificity:
            specificity = self._calculate_specificity(query, vocabulary_size)
            difficulty_scores.append(specificity)
            weights.append(0.2)

        # 加权平均
        if difficulty_scores:
            weights = np.array(weights)
            weights = weights / weights.sum()  # 归一化
            difficulty = np.average(difficulty_scores, weights=weights)
        else:
            # 如果没有特征可用，返回中等难度
            difficulty = 0.5

        logging.debug(f"Query difficulty: {difficulty:.3f}")
        return float(difficulty)

    def _calculate_semantic_uncertainty(
        self,
        similarities: List[float]
    ) -> float:
        """
        计算语义不确定性

        原理：
        - 如果top-k个结果的相似度分数差异很大（标准差大），说明查询模糊
        - 如果top-1和top-2的分数接近，说明难以判断哪个更相关

        Args:
            similarities (List[float]): 检索结果的相似度分数

        Returns:
            float: 不确定性分数 [0, 1]
        """
        if not similarities or len(similarities) < 2:
            return 0.5

        similarities = np.array(similarities)

        # 指标1：分数的标准差（归一化）
        std_score = np.std(similarities)
        std_normalized = min(std_score / 0.3, 1.0)  # 假设0.3是高标准差

        # 指标2：top-1和top-2的差距
        if len(similarities) >= 2:
            gap = abs(similarities[0] - similarities[1])
            # 差距越小，不确定性越高
            gap_score = 1.0 - min(gap / 0.2, 1.0)  # 假设0.2是显著差距
        else:
            gap_score = 0.5

        # 综合不确定性
        uncertainty = 0.6 * std_normalized + 0.4 * gap_score
        return float(uncertainty)

    def _calculate_length_complexity(self, query: str) -> float:
        """
        计算查询长度复杂度

        原理：
        - 过短的查询（<5词）：信息不足，困难
        - 适中的查询（5-15词）：信息充分，简单
        - 过长的查询（>15词）：可能包含多个意图，困难

        Args:
            query (str): 查询文本

        Returns:
            float: 长度复杂度 [0, 1]
        """
        words = query.split()
        word_count = len(words)

        # 最佳长度范围：5-15个词
        if 5 <= word_count <= 15:
            # 在最佳范围内，复杂度低
            complexity = 0.2
        elif word_count < 5:
            # 过短，复杂度随长度减少而增加
            complexity = 0.5 + (5 - word_count) * 0.1
            complexity = min(complexity, 1.0)
        else:
            # 过长，复杂度随长度增加而增加
            complexity = 0.3 + (word_count - 15) * 0.05
            complexity = min(complexity, 1.0)

        return complexity

    def _calculate_specificity(
        self,
        query: str,
        vocabulary_size: Optional[int] = None
    ) -> float:
        """
        计算查询特异性

        原理：
        - 使用常见词（如"good", "bad"）的查询：特异性低，困难
        - 使用专业词汇或稀有词的查询：特异性高，简单

        Args:
            query (str): 查询文本
            vocabulary_size (int, optional): 语料库词汇量

        Returns:
            float: 特异性分数 [0, 1]，低表示更困难
        """
        # 常见停用词列表（简化版）
        common_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'can', 'could', 'should', 'may', 'might', 'must',
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'this', 'that', 'these', 'those',
            'good', 'bad', 'nice', 'great', 'very', 'really'
        }

        words = query.lower().split()
        if not words:
            return 0.5

        # 计算非常见词的比例
        specific_words = [w for w in words if w not in common_words]
        specificity_ratio = len(specific_words) / len(words)

        # 特异性高 -> 难度低
        difficulty = 1.0 - specificity_ratio
        return difficulty


class AdaptiveRetriever:
    """
    自适应检索器

    根据查询难度动态调整检索策略
    """

    def __init__(
        self,
        difficulty_estimator: Optional[DifficultyEstimator] = None,
        strategies: Optional[Dict[str, RetrievalStrategy]] = None
    ):
        """
        初始化自适应检索器

        Args:
            difficulty_estimator (DifficultyEstimator, optional): 难度评估器
            strategies (Dict[str, RetrievalStrategy], optional): 检索策略配置
                键为难度级别: 'easy', 'medium', 'hard'
        """
        self.difficulty_estimator = (
            difficulty_estimator if difficulty_estimator
            else DifficultyEstimator()
        )

        # 默认策略配置
        self.strategies = strategies if strategies else {
            'easy': RetrievalStrategy(
                k=3,                      # 少量检索
                use_rerank=False,         # 不使用重排序
                confidence_threshold=0.7   # 高置信度阈值
            ),
            'medium': RetrievalStrategy(
                k=5,                      # 中等数量
                use_rerank=True,          # 使用重排序
                confidence_threshold=0.5
            ),
            'hard': RetrievalStrategy(
                k=10,                     # 大量检索
                use_rerank=True,          # 使用重排序
                confidence_threshold=0.3   # 低置信度阈值
            )
        }

        logging.info("AdaptiveRetriever initialized with strategies: "
                     f"easy(k={self.strategies['easy'].k}), "
                     f"medium(k={self.strategies['medium'].k}), "
                     f"hard(k={self.strategies['hard'].k})")

    # ==================== 已弃用：带探测的检索方法 ====================
    # 注意：以下方法已被 estimate_difficulty_without_retrieval() 替代
    # 保留代码以备将来需要带探测的检索策略时使用

    # def retrieve(
    #     self,
    #     query: str,
    #     retrieval_function: Callable,
    #     initial_k: int = 10,
    #     **retrieval_kwargs
    # ) -> Tuple[List[Dict], Dict[str, any]]:
    #     """
    #     执行自适应检索（已弃用 - 使用 estimate_difficulty_without_retrieval 代替）
    #
    #     Args:
    #         query (str): 查询文本
    #         retrieval_function (Callable): 检索函数
    #             应接受query和k参数，返回带有similarity字段的候选文档列表
    #         initial_k (int): 初始探测检索数量（用于评估难度）
    #         **retrieval_kwargs: 传递给retrieval_function的额外参数
    #
    #     Returns:
    #         Tuple[List[Dict], Dict]: (检索结果, 元信息)
    #             元信息包含: difficulty, strategy_used, actual_k等
    #
    #     工作流程：
    #         1. 执行小规模探测检索
    #         2. 根据结果评估查询难度
    #         3. 选择相应的检索策略
    #         4. 执行最终检索
    #     """
    #     # 步骤1：探测检索（probe retrieval）
    #     logging.info(f"Step 1: Probing with k={initial_k}...")
    #     probe_results = retrieval_function(
    #         query, k=initial_k, **retrieval_kwargs
    #     )
    #
    #     # 步骤2：提取相似度分数并评估难度
    #     similarities = [
    #         r.get('similarity', 0.0) for r in probe_results
    #     ]
    #     difficulty = self.difficulty_estimator.estimate(
    #         query=query,
    #         top_similarities=similarities
    #     )
    #
    #     # 步骤3：选择策略
    #     strategy = self._select_strategy(difficulty)
    #     logging.info(f"Step 2: Difficulty={difficulty:.3f}, "
    #                  f"Strategy={self._get_difficulty_level(difficulty)}, "
    #                  f"k={strategy.k}")
    #
    #     # 步骤4：执行最终检索
    #     if strategy.k <= initial_k:
    #         # 探测检索已经足够，直接使用
    #         final_results = probe_results[:strategy.k]
    #         logging.info("Using probe results (sufficient)")
    #     else:
    #         # 需要更多结果，重新检索
    #         logging.info(f"Step 3: Retrieving with k={strategy.k}...")
    #         final_results = retrieval_function(
    #             query, k=strategy.k, **retrieval_kwargs
    #         )
    #
    #     # 元信息
    #     metadata = {
    #         'difficulty': difficulty,
    #         'difficulty_level': self._get_difficulty_level(difficulty),
    #         'strategy_used': {
    #             'k': strategy.k,
    #             'use_rerank': strategy.use_rerank,
    #             'confidence_threshold': strategy.confidence_threshold
    #         },
    #         'actual_retrieved': len(final_results),
    #         'avg_similarity': np.mean(similarities) if similarities else 0.0,
    #         'similarity_std': np.std(similarities) if similarities else 0.0
    #     }
    #
    #     return final_results, metadata

    # def _select_strategy(self, difficulty: float) -> RetrievalStrategy:
    #     """
    #     根据难度分数选择检索策略（已弃用）
    #
    #     Args:
    #         difficulty (float): 难度分数 [0, 1]
    #
    #     Returns:
    #         RetrievalStrategy: 选定的检索策略
    #     """
    #     if difficulty < 0.35:
    #         return self.strategies['easy']
    #     elif difficulty < 0.65:
    #         return self.strategies['medium']
    #     else:
    #         return self.strategies['hard']

    def _get_difficulty_level(self, difficulty: float) -> str:
        """获取难度级别的字符串表示"""
        if difficulty < 0.35:
            return 'easy'
        elif difficulty < 0.65:
            return 'medium'
        else:
            return 'hard'

    def estimate_difficulty_without_retrieval(
        self,
        query: str
    ) -> Tuple[float, int, str]:
        """
        仅根据query文本评估难度，不需要探测性检索

        这个方法用于在检索前就决定召回数量，避免探测性检索的开销。
        难度评估基于query的文本特征（长度、特异性等），不依赖检索结果。

        Args:
            query (str): 查询文本

        Returns:
            Tuple[float, int, str]: (难度分数, 建议召回数量, 难度级别)
                - difficulty: 难度分数，范围[0, 1]
                - recall_k: 建议的召回数量
                - difficulty_level: 难度级别字符串 ('easy', 'medium', 'hard')

        召回数量策略：
            - 简单查询 (difficulty < 0.35): recall_k = 20
                理由：简单查询相关文档集中，少量召回即可覆盖
            - 中等查询 (0.35 <= difficulty < 0.65): recall_k = 50
                理由：需要更多候选以保证覆盖率
            - 困难查询 (difficulty >= 0.65): recall_k = 100
                理由：困难查询需要大量候选，让MMR从中筛选多样化结果

        学术依据：
            Query Performance Prediction (QPP) 研究表明，可以通过query特征
            预测检索难度，从而动态调整检索策略。
            参考：Cronen-Townsend et al. (2002), "Predicting Query Performance"
        """
        # 仅用query文本评估难度（不使用检索结果）
        difficulty = self.difficulty_estimator.estimate(
            query=query,
            top_similarities=None  # 关键：不使用检索结果
        )

        # 根据难度决定召回数量
        if difficulty < 0.35:  # 简单查询
            recall_k = 20
            difficulty_level = 'easy'
        elif difficulty < 0.65:  # 中等查询
            recall_k = 50
            difficulty_level = 'medium'
        else:  # 困难查询
            recall_k = 100
            difficulty_level = 'hard'

        logging.info(
            f"Difficulty estimation (no retrieval): "
            f"difficulty={difficulty:.3f}, level={difficulty_level}, recall_k={recall_k}"
        )

        return difficulty, recall_k, difficulty_level

    def estimate_difficulty_with_lightweight_probe(
        self,
        query: str,
        retrieval_function: Callable,
        probe_k: int = 5,
        **retrieval_kwargs
    ) -> Tuple[float, int, str, List[Dict]]:
        """
        使用轻量级探测评估难度（包含语义不确定性维度）

        此方法通过小规模探测检索（默认5个文档）来获取语义不确定性信息，
        从而使用完整的3个维度进行难度评估：
        1. 语义不确定性 (50%权重) - 需要探测检索结果
        2. 查询长度复杂度 (30%权重) - 基于文本特征
        3. 查询特异性 (20%权重) - 基于文本特征

        Args:
            query (str): 查询文本
            retrieval_function (Callable): 检索函数，接受query和k参数
            probe_k (int): 探测检索数量，默认5个（平衡准确性和效率）
            **retrieval_kwargs: 传递给retrieval_function的额外参数

        Returns:
            Tuple[float, int, str, List[Dict]]: (难度分数, 建议召回数量, 难度级别, 探测结果)
                - difficulty: 难度分数，范围[0, 1]
                - recall_k: 建议的召回数量 (20/50/100)
                - difficulty_level: 难度级别 ('easy', 'medium', 'hard')
                - probe_results: 探测检索的结果（可能直接使用，避免重复检索）

        召回数量策略：
            - 简单查询 (difficulty < 0.35):
                - 如果probe_k=5: 直接使用探测结果
                - 否则: recall_k = 20
            - 中等查询 (0.35 <= difficulty < 0.65): recall_k = 50
            - 困难查询 (difficulty >= 0.65): recall_k = 100

        学术依据：
            - 完整3维度评估提供更准确的难度预测
            - 轻量级探测（k=5）成本低，不影响效率
            - 参考: Cronen-Townsend et al. (2002), "Predicting Query Performance"
        """
        # 步骤1：轻量级探测检索
        logging.info(f"Lightweight probe: retrieving {probe_k} documents...")
        probe_results = retrieval_function(query, k=probe_k, **retrieval_kwargs)

        # 步骤2：提取相似度分数
        similarities = [
            r.get('similarity', 0.0) for r in probe_results
        ]

        # 步骤3：使用3个维度评估难度
        difficulty = self.difficulty_estimator.estimate(
            query=query,
            top_similarities=similarities  # 关键：提供相似度，启用语义不确定性
        )

        # 步骤4：根据难度决定召回数量
        # 注意：为了给MMR足够的候选进行多样性优化，即使简单查询也召回至少20个
        if difficulty < 0.35:  # 简单查询
            recall_k = 20  # 足够MMR选择10个多样化文档
            difficulty_level = 'easy'
        elif difficulty < 0.65:  # 中等查询
            recall_k = 50  # MMR有充足的选择空间
            difficulty_level = 'medium'
        else:  # 困难查询
            recall_k = 100  # MMR从大量候选中筛选多角度信息
            difficulty_level = 'hard'

        logging.info(
            f"Difficulty estimation (with probe): "
            f"difficulty={difficulty:.3f}, level={difficulty_level}, recall_k={recall_k}"
        )

        return difficulty, recall_k, difficulty_level, probe_results

    # def should_use_rerank(self, difficulty: float) -> bool:
    #     """
    #     判断是否应该使用重排序（已弃用）
    #
    #     Args:
    #         difficulty (float): 难度分数
    #
    #     Returns:
    #         bool: 是否使用重排序
    #     """
    #     strategy = self._select_strategy(difficulty)
    #     return strategy.use_rerank


# ==================== 使用示例（已弃用） ====================
# 注意：以下示例代码适用于旧的 retrieve() 方法
# 新方法的使用示例请参考 test_adaptive_no_retrieval.py

# def example_usage():
#     """
#     展示如何使用AdaptiveRetriever（已弃用）
#     """
#
#     # 模拟检索函数
#     def mock_retrieval(query: str, k: int) -> List[Dict]:
#         """
#         模拟的检索函数
#         实际使用时，替换为你的真实检索逻辑
#         """
#         # 模拟不同难度的查询返回不同的相似度分布
#         if "simple" in query.lower():
#             # 简单查询：高相似度，小方差
#             similarities = np.random.normal(0.85, 0.05, k)
#         elif "complex" in query.lower():
#             # 复杂查询：低相似度，大方差
#             similarities = np.random.normal(0.50, 0.15, k)
#         else:
#             # 中等查询
#             similarities = np.random.normal(0.70, 0.10, k)
#
#         similarities = np.clip(similarities, 0, 1)
#
#         results = []
#         for i, sim in enumerate(similarities):
#             results.append({
#                 'id': i,
#                 'text': f"Document {i} for query: {query}",
#                 'similarity': float(sim)
#             })
#
#         return results
#
#     # 测试不同难度的查询
#     test_queries = [
#         ("This is a simple and clear query", "简单查询"),
#         ("What about this medium difficulty query with some context", "中等查询"),
#         ("This is a complex and ambiguous query with multiple potential interpretations", "复杂查询")
#     ]
#
#     retriever = AdaptiveRetriever()
#
#     print("=" * 80)
#     print("自适应检索示例")
#     print("=" * 80)
#
#     for query, description in test_queries:
#         print(f"\n{'='*80}")
#         print(f"查询类型: {description}")
#         print(f"查询文本: {query}")
#         print(f"{'='*80}")
#
#         results, metadata = retriever.retrieve(
#             query=query,
#             retrieval_function=mock_retrieval,
#             initial_k=10
#         )
#
#         print(f"\n难度评估:")
#         print(f"  - 难度分数: {metadata['difficulty']:.3f}")
#         print(f"  - 难度级别: {metadata['difficulty_level']}")
#         print(f"  - 平均相似度: {metadata['avg_similarity']:.3f}")
#         print(f"  - 相似度标准差: {metadata['similarity_std']:.3f}")
#
#         print(f"\n策略选择:")
#         print(f"  - 检索数量 k: {metadata['strategy_used']['k']}")
#         print(f"  - 使用重排序: {metadata['strategy_used']['use_rerank']}")
#         print(f"  - 置信度阈值: {metadata['strategy_used']['confidence_threshold']}")
#
#         print(f"\n检索结果: (返回 {len(results)} 个)")
#         for i, result in enumerate(results[:3], 1):  # 只显示前3个
#             print(f"  {i}. 相似度: {result['similarity']:.3f} - {result['text'][:50]}...")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 运行示例（已弃用 - 使用 test_adaptive_no_retrieval.py）
    # example_usage()
    print("此模块的示例代码已弃用，请运行 test_adaptive_no_retrieval.py 查看新方法的使用示例")

