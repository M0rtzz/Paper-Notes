---
title: >-
  ICML2026 推荐系统方向1篇论文解读
description: >-
  1篇ICML2026的推荐系统方向论文解读，涵盖推荐系统等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "推荐系统"
  - "论文解读"
  - "论文笔记"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎁 推荐系统

**🧪 ICML2026** · **1** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (19)](../../ACL2026/recommender/index.md) · [🔬 ICLR2026 (10)](../../ICLR2026/recommender/index.md) · [🤖 AAAI2026 (26)](../../AAAI2026/recommender/index.md) · [🧠 NeurIPS2025 (24)](../../NeurIPS2025/recommender/index.md) · [🧪 ICML2025 (17)](../../ICML2025/recommender/index.md) · [💬 ACL2025 (7)](../../ACL2025/recommender/index.md)

**[Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control](can_recommender_systems_teach_themselves_a_recursive_self-improving_framework_wi.md)**

:   RSIR 让序列推荐模型用自身预测能力生成新的合成用户交互序列、再训练一个新模型，并用基于排名的"保真度检查"过滤掉偏离用户偏好流形的样本，防止 self-consuming model collapse；在 4 个数据集 × 3 个主流 backbone 上稳定提升 NDCG/Recall 4–11%，并理论上证明该过程等价于沿用户偏好流形切空间的隐式正则化。
