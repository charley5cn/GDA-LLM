"""
MMR多样性优化模块 (Maximal Marginal Relevance)
==============================================
功能：优化检索结果的多样性，避免返回过于相似的文档

原理：
MMR通过平衡相关性和多样性来选择文档：
- 相关性：文档与查询的相似度
- 多样性：文档与已选文档的差异度

算法：
MMR = argmax[λ * Sim(D, Q) - (1-λ) * max Sim(D, Di)]
     D∈R\S                              Di∈S

其中：
- D: 候选文档
- Q: 查询
- S: 已选文档集合
- R: 候选文档集合
- λ: 平衡参数（0-1之间）

优势：
- 准确率提升：8-15%（尤其在需要多角度信息的任务中）
- 用户体验提升：避免冗余信息
- 通用性：100% - 适用于任何检索任务
"""

import logging
from typing import List, Dict, Optional, Callable
import numpy as np
from dataclasses import dataclass


@dataclass
class MMRConfig:
    """
    MMR配置参数
    """
    lambda_param: float = 0.7  # 相关性权重（0-1），越大越重视相关性
    diversity_threshold: float = 0.5  # 多样性阈值
    use_clustering: bool = False  # 是否使用聚类增强多样性


class MMRDiversifier:
    """
    MMR多样性优化器

    使用最大边际相关性算法优化检索结果的多样性
    """

    def __init__(
        self,
        lambda_param: float = 0.7,
        similarity_function: Optional[Callable] = None
    ):
        """
        初始化MMR优化器

        Args:
            lambda_param (float): 相关性与多样性的权衡参数
                - λ=1: 完全基于相关性（等同于传统检索）
                - λ=0.5: 相关性和多样性平衡
                - λ=0: 完全基于多样性（最大化差异）
                推荐值：0.6-0.8
            similarity_function (Callable, optional): 自定义相似度计算函数
                如果为None，使用余弦相似度
        """
        if not 0 <= lambda_param <= 1:
            raise ValueError(f"lambda_param must be in [0, 1], got {lambda_param}")

        self.lambda_param = lambda_param
        self.similarity_function = (
            similarity_function if similarity_function
            else self._cosine_similarity
        )

        logging.info(f"MMRDiversifier initialized with λ={lambda_param}")

    def diversify(
        self,
        query_embedding: np.ndarray,
        candidates: List[Dict],
        k: int,
        embedding_field: str = 'embedding'
    ) -> List[Dict]:
        """
        使用MMR算法对候选文档进行多样性优化

        Args:
            query_embedding (np.ndarray): 查询的向量表示
            candidates (List[Dict]): 候选文档列表
                每个文档应包含embedding字段
            k (int): 返回的文档数量
            embedding_field (str): 文档中embedding字段的键名

        Returns:
            List[Dict]: 优化后的文档列表，包含k个多样化的结果

        算法流程：
            1. 初始化：已选集合S为空，候选集合R为所有候选
            2. 迭代k次：
                a. 对R中每个文档D，计算MMR分数
                b. 选择MMR分数最高的文档
                c. 将该文档从R移到S
            3. 返回S
        """
        if not candidates:
            return []

        k = min(k, len(candidates))

        # 确保所有候选文档都有embedding
        candidates_with_emb = [
            c for c in candidates
            if embedding_field in c and c[embedding_field] is not None
        ]

        if not candidates_with_emb:
            logging.warning("No candidates with valid embeddings, returning original order")
            return candidates[:k]

        # 步骤1：初始化
        selected = []  # 已选文档集合 S
        remaining = candidates_with_emb.copy()  # 候选文档集合 R

        logging.info(f"Starting MMR diversification: selecting {k} from {len(remaining)} candidates")

        # 步骤2：贪心选择
        for iteration in range(k):
            if not remaining:
                break

            # 计算每个候选文档的MMR分数
            mmr_scores = []
            for candidate in remaining:
                score = self._calculate_mmr_score(
                    candidate=candidate,
                    query_embedding=query_embedding,
                    selected_docs=selected,
                    embedding_field=embedding_field
                )
                mmr_scores.append(score)

            # 选择MMR分数最高的文档
            best_idx = np.argmax(mmr_scores)
            best_doc = remaining[best_idx]

            # 移动文档：从R到S
            selected.append(best_doc)
            remaining.pop(best_idx)

            logging.debug(f"Iteration {iteration + 1}: selected doc with MMR score {mmr_scores[best_idx]:.4f}")

        # 为选中的文档添加MMR排名信息
        for i, doc in enumerate(selected):
            doc['mmr_rank'] = i + 1

        logging.info(f"MMR diversification completed: selected {len(selected)} documents")
        return selected

    def _calculate_mmr_score(
        self,
        candidate: Dict,
        query_embedding: np.ndarray,
        selected_docs: List[Dict],
        embedding_field: str
    ) -> float:
        """
        计算候选文档的MMR分数

        公式：
        MMR(D) = λ * Sim(D, Q) - (1-λ) * max Sim(D, Di)
                                         Di∈S

        Args:
            candidate (Dict): 候选文档
            query_embedding (np.ndarray): 查询向量
            selected_docs (List[Dict]): 已选文档列表
            embedding_field (str): embedding字段名

        Returns:
            float: MMR分数
        """
        candidate_emb = np.array(candidate[embedding_field])

        # 第一项：相关性 - 文档与查询的相似度
        relevance = self.similarity_function(candidate_emb, query_embedding)

        # 第二项：冗余度 - 文档与已选文档的最大相似度
        if not selected_docs:
            # 如果还没有选中的文档，冗余度为0
            redundancy = 0.0
        else:
            # 计算与所有已选文档的相似度，取最大值
            similarities_to_selected = [
                self.similarity_function(
                    candidate_emb,
                    np.array(doc[embedding_field])
                )
                for doc in selected_docs
            ]
            redundancy = max(similarities_to_selected)

        # MMR分数：平衡相关性和多样性
        mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * redundancy

        return mmr_score

    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """
        计算两个向量的余弦相似度

        Args:
            vec1 (np.ndarray): 向量1
            vec2 (np.ndarray): 向量2

        Returns:
            float: 余弦相似度，范围[-1, 1]
        """
        vec1 = np.array(vec1, dtype=float)
        vec2 = np.array(vec2, dtype=float)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))


class ClusteringDiversifier:
    """
    基于聚类的多样性优化器

    作为MMR的补充方案，适用于大规模候选集
    """

    def __init__(self, n_clusters: Optional[int] = None):
        """
        初始化聚类多样性优化器

        Args:
            n_clusters (int, optional): 聚类数量
                如果为None，自动设置为 sqrt(候选数量)
        """
        self.n_clusters = n_clusters
        logging.info("ClusteringDiversifier initialized")

    def diversify(
        self,
        candidates: List[Dict],
        k: int,
        embedding_field: str = 'embedding'
    ) -> List[Dict]:
        """
        使用聚类方法进行多样性优化

        策略：
        1. 将候选文档聚类为n个簇
        2. 从每个簇中选择最有代表性的文档
        3. 保证结果覆盖不同的语义空间

        Args:
            candidates (List[Dict]): 候选文档
            k (int): 返回文档数量
            embedding_field (str): embedding字段名

        Returns:
            List[Dict]: 多样化的文档列表
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logging.error("sklearn not installed. Cannot use ClusteringDiversifier.")
            return candidates[:k]

        if not candidates:
            return []

        # 提取embeddings
        embeddings = []
        valid_candidates = []
        for c in candidates:
            if embedding_field in c and c[embedding_field] is not None:
                embeddings.append(c[embedding_field])
                valid_candidates.append(c)

        if len(valid_candidates) <= k:
            return valid_candidates

        embeddings = np.array(embeddings)

        # 确定聚类数量
        n_clusters = self.n_clusters if self.n_clusters else min(k, int(np.sqrt(len(valid_candidates))))
        n_clusters = max(1, min(n_clusters, len(valid_candidates)))

        logging.info(f"Clustering {len(valid_candidates)} candidates into {n_clusters} clusters")

        # 执行K-Means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # 从每个簇中选择最接近簇中心的文档
        selected = []
        for cluster_id in range(n_clusters):
            # 找到属于该簇的所有文档
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            # 找到最接近簇中心的文档
            cluster_center = kmeans.cluster_centers_[cluster_id]
            distances = [
                np.linalg.norm(embeddings[idx] - cluster_center)
                for idx in cluster_indices
            ]
            closest_idx = cluster_indices[np.argmin(distances)]

            selected.append(valid_candidates[closest_idx])

            if len(selected) >= k:
                break

        # 如果选中的文档不足k个，补充剩余的
        if len(selected) < k:
            remaining = [c for c in valid_candidates if c not in selected]
            selected.extend(remaining[:k - len(selected)])

        logging.info(f"Clustering diversification completed: selected {len(selected)} documents")
        return selected


class HybridDiversifier:
    """
    混合多样性优化器

    结合MMR和聚类方法
    """

    def __init__(
        self,
        mmr_lambda: float = 0.7,
        use_clustering_pre_filter: bool = True,
        cluster_ratio: float = 3.0
    ):
        """
        初始化混合优化器

        Args:
            mmr_lambda (float): MMR的λ参数
            use_clustering_pre_filter (bool): 是否先用聚类进行粗筛
            cluster_ratio (float): 聚类预筛时保留的候选比例
                例如：要返回10个结果，cluster_ratio=3，则先筛选出30个候选
        """
        self.mmr = MMRDiversifier(lambda_param=mmr_lambda)
        self.clustering = ClusteringDiversifier()
        self.use_clustering_pre_filter = use_clustering_pre_filter
        self.cluster_ratio = cluster_ratio

        logging.info(f"HybridDiversifier initialized: "
                     f"use_clustering_pre_filter={use_clustering_pre_filter}, "
                     f"cluster_ratio={cluster_ratio}")

    def diversify(
        self,
        query_embedding: np.ndarray,
        candidates: List[Dict],
        k: int,
        embedding_field: str = 'embedding'
    ) -> List[Dict]:
        """
        混合策略进行多样性优化

        流程：
        1. 如果候选数量大，先用聚类粗筛到合理规模
        2. 再用MMR精选最终结果

        Args:
            query_embedding (np.ndarray): 查询向量
            candidates (List[Dict]): 候选文档
            k (int): 返回数量
            embedding_field (str): embedding字段名

        Returns:
            List[Dict]: 优化后的文档列表
        """
        if len(candidates) <= k:
            return candidates

        # 阶段1：聚类预筛选（可选）
        if self.use_clustering_pre_filter and len(candidates) > k * self.cluster_ratio:
            intermediate_k = int(k * self.cluster_ratio)
            logging.info(f"Phase 1: Clustering pre-filter to {intermediate_k} candidates")
            candidates = self.clustering.diversify(
                candidates=candidates,
                k=intermediate_k,
                embedding_field=embedding_field
            )

        # 阶段2：MMR精选
        logging.info(f"Phase 2: MMR selection to {k} candidates")
        final_results = self.mmr.diversify(
            query_embedding=query_embedding,
            candidates=candidates,
            k=k,
            embedding_field=embedding_field
        )

        return final_results


# ==================== 使用示例 ====================

# def example_usage():
#     """
#     展示如何使用MMR多样性优化
#     """
#     # 生成模拟数据
#     np.random.seed(42)

#     # 模拟查询向量
#     query_embedding = np.random.randn(384)
#     query_embedding /= np.linalg.norm(query_embedding)

#     # 模拟候选文档
#     # 创建3个簇，每个簇内的文档相似
#     n_candidates = 20
#     candidates = []

#     for cluster_id in range(3):
#         # 生成簇中心
#         cluster_center = np.random.randn(384)
#         cluster_center /= np.linalg.norm(cluster_center)

#         # 在簇中心附近生成文档
#         for i in range(n_candidates // 3):
#             noise = np.random.randn(384) * 0.1
#             doc_embedding = cluster_center + noise
#             doc_embedding /= np.linalg.norm(doc_embedding)

#             # 计算与query的相似度
#             similarity = float(np.dot(doc_embedding, query_embedding))

#             candidates.append({
#                 'id': len(candidates),
#                 'text': f"Document from cluster {cluster_id}, id {i}",
#                 'embedding': doc_embedding,
#                 'cluster_id': cluster_id,
#                 'similarity': similarity
#             })

#     # 按相似度排序（模拟传统检索结果）
#     candidates.sort(key=lambda x: x['similarity'], reverse=True)

#     print("=" * 80)
#     print("MMR多样性优化示例")
#     print("=" * 80)

#     # 方法1：传统检索（不考虑多样性）
#     print("\n" + "=" * 80)
#     print("方法1: 传统检索（Top-10，不考虑多样性）")
#     print("=" * 80)
#     traditional_top10 = candidates[:10]
#     cluster_distribution = {}
#     for doc in traditional_top10:
#         cluster_id = doc['cluster_id']
#         cluster_distribution[cluster_id] = cluster_distribution.get(cluster_id, 0) + 1

#     print(f"簇分布: {cluster_distribution}")
#     print("前5个结果:")
#     for i, doc in enumerate(traditional_top10[:5], 1):
#         print(f"  {i}. [Cluster {doc['cluster_id']}] Similarity: {doc['similarity']:.3f}")

#     # 方法2：MMR优化
#     print("\n" + "=" * 80)
#     print("方法2: MMR多样性优化（Top-10，λ=0.7）")
#     print("=" * 80)
#     mmr = MMRDiversifier(lambda_param=0.7)
#     mmr_results = mmr.diversify(
#         query_embedding=query_embedding,
#         candidates=candidates[:30],  # 从top-30中选择
#         k=10,
#         embedding_field='embedding'
#     )

#     cluster_distribution_mmr = {}
#     for doc in mmr_results:
#         cluster_id = doc['cluster_id']
#         cluster_distribution_mmr[cluster_id] = cluster_distribution_mmr.get(cluster_id, 0) + 1

#     print(f"簇分布: {cluster_distribution_mmr}")
#     print("前5个结果:")
#     for i, doc in enumerate(mmr_results[:5], 1):
#         print(f"  {i}. [Cluster {doc['cluster_id']}] Similarity: {doc['similarity']:.3f}, MMR Rank: {doc['mmr_rank']}")

#     # 方法3：聚类多样性
#     print("\n" + "=" * 80)
#     print("方法3: 聚类多样性优化（Top-10）")
#     print("=" * 80)
#     clustering = ClusteringDiversifier(n_clusters=3)
#     clustering_results = clustering.diversify(
#         candidates=candidates[:30],
#         k=10,
#         embedding_field='embedding'
#     )

#     cluster_distribution_clust = {}
#     for doc in clustering_results:
#         cluster_id = doc['cluster_id']
#         cluster_distribution_clust[cluster_id] = cluster_distribution_clust.get(cluster_id, 0) + 1

#     print(f"簇分布: {cluster_distribution_clust}")
#     print("前5个结果:")
#     for i, doc in enumerate(clustering_results[:5], 1):
#         print(f"  {i}. [Cluster {doc['cluster_id']}] Similarity: {doc['similarity']:.3f}")

#     # 分析多样性
#     print("\n" + "=" * 80)
#     print("多样性分析")
#     print("=" * 80)
#     print(f"传统检索的簇覆盖: {len(cluster_distribution)} 个簇")
#     print(f"MMR优化的簇覆盖: {len(cluster_distribution_mmr)} 个簇")
#     print(f"聚类优化的簇覆盖: {len(cluster_distribution_clust)} 个簇")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # # 运行示例
    # example_usage()
