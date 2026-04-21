---
title: >-
  [论文解读] vCache: Verified Semantic Prompt Caching
description: >-
  [ICLR2026][Semantic Caching] 提出 vCache——首个具有**用户定义错误率保证**的语义缓存系统，通过在线学习为每个缓存嵌入独立估计最优相似度阈值，无需预训练即可在满足正确性约束下实现最高 12.5× 缓存命中率提升和 26× 错误率降低。
tags:
  - ICLR2026
  - Semantic Caching
  - LLM Inference Optimization
  - Error-Rate Guarantee
  - online learning
  - Per-Embedding Threshold
---

# vCache: Verified Semantic Prompt Caching

**会议**: ICLR2026  
**arXiv**: [2502.03771](https://arxiv.org/abs/2502.03771)  
**代码**: [GitHub](https://github.com/vcache-project/vCache) | [Benchmarks](https://huggingface.co/vCache)  
**领域**: llm_nlp  
**关键词**: Semantic Caching, LLM Inference Optimization, Error-Rate Guarantee, online learning, Per-Embedding Threshold  
**作者**: Luis Gaspar Schroeder, Aditya Desai, Alejandro Cuadron, Kyle Chu, Shu Liu, Mark Zhao, Stephan Krusche, Alfons Kemper, Matei Zaharia, Joseph E. Gonzalez（UC Berkeley, TU Munich, ETH Zurich, Stanford）

## 一句话总结

提出 vCache——首个具有**用户定义错误率保证**的语义缓存系统，通过在线学习为每个缓存嵌入独立估计最优相似度阈值，无需预训练即可在满足正确性约束下实现最高 12.5× 缓存命中率提升和 26× 错误率降低。

## 研究背景与动机

**LLM 推理成本高**：每次 prompt 需要多次前向传播，部署昂贵且延迟大。

**语义缓存的价值**：对"加拿大首都是哪里？"已有缓存时，"Which city is Canada's capital?" 应复用同一回复，最高可降低 100× 延迟。

**静态阈值的根本缺陷**：
   - 现有系统（GPTCache、Azure、AWS 等）使用**全局固定阈值** $t$，对所有 prompt 一视同仁
   - 但实验发现（Figure 3）：正确和错误缓存命中的相似度分布**高度重叠**
   - **每个嵌入的最优阈值差异极大**，单一阈值不可能同时兼顾低错误率和高命中率

**无错误率保证**：现有系统无法保证返回的缓存回复的正确性，在生产环境中难以被信任

**嵌入微调的局限**：需要监督训练、仅限开源模型、对 OOD 数据泛化差

## 核心贡献

1. 首个提供**用户定义错误率保证**的可验证语义缓存
2. **在线阈值学习算法**：无需预训练，为每个缓存嵌入独立学习阈值，与嵌入模型无关
3. 证明**动态逐嵌入阈值**优于静态阈值和微调嵌入
4. 开源 4 个语义缓存基准（LMArena/Classification/SearchQueries/Combo）

## 方法详解

### 系统概览

语义缓存的基本流程：
1. 将 prompt $x$ 嵌入为 $\mathcal{E}(x) \in \mathbb{R}^d$
2. 从向量数据库检索最近邻 $\text{nn}(x) = \arg\max_{y \in C} \text{sim}(\mathcal{E}(x), \mathcal{E}(y))$
3. 计算相似度 $s(x) = \text{sim}(\mathcal{E}(x), \mathcal{E}(\text{nn}(x)))$
4. 决策：若 $s(x) \ge t$ 则**exploit**（返回缓存回复），否则**explore**（调用 LLM）

### 数据模型

缓存存储三元组：

$$\mathcal{D} = \{(\mathcal{E}(x_i), r(x_i), \mathcal{O}(x_i))\}_{i=0}^{n}$$

其中每个嵌入的元数据 $\mathcal{O}(x_i)$ 记录了所有以 $x_i$ 为最近邻的后续 prompt 的相似度和匹配信息：

$$\mathcal{O}(x_i) = \{(s(x_j), c(x_j)) \mid \text{nn}(x_j) = x_i\}_{j=i+1}^{n}$$

$$c(x) = \begin{cases} 1 & \text{if } r(\text{nn}(x)) = r(x) \\ 0 & \text{otherwise} \end{cases}$$

### 用户保证

用户指定最大错误率 $\delta$，vCache 保证：

$$\Pr(\mathbf{vCache}(x) = r(x)) \ge 1 - \delta$$

分解正确概率为两个不相交事件：

$$\Pr(\mathbf{vCache}(x) = r(x)) = \Pr(\text{explore} \mid x, \mathcal{D}) + (1 - \Pr(\text{explore} \mid x, \mathcal{D})) \cdot \Pr(c(x) = 1 \mid x, \mathcal{D})$$

由此推导出**探索概率下界**：

$$\Pr(\text{explore} \mid x, \mathcal{D}) \ge \frac{(1-\delta) - \Pr(c(x)=1 \mid x, \mathcal{D})}{1 - \Pr(c(x)=1 \mid x, \mathcal{D})} = \tau_{\text{nn}(x)}(s(x))$$

### sigmoid 参数化建模

对每个嵌入，使用 sigmoid 函数建模相似度与正确性的关系：

$$\Pr(c(x) = 1 \mid x, \mathcal{D}) = \mathcal{L}(s(x), t, \gamma) = \frac{1}{1 + e^{-\gamma(s(x) - t)}}$$

其中 $t$ 是嵌入特定的决策边界，$\gamma$ 控制陡峭度。通过**最大似然估计**（二元交叉熵损失）求解：

$$\hat{t}, \hat{\gamma} = \arg\min_{t, \gamma} \mathbb{E}_{(s,c) \in \mathcal{O}_{\text{nn}(x)}} \left[ c \cdot \log(\mathcal{L}(s, t, \gamma)) + (1-c) \cdot \log(1 - \mathcal{L}(s, t, \gamma)) \right]$$

### 置信区间校准

因样本有限，直接用 $\hat{t}, \hat{\gamma}$ 无法保证精确。vCache 使用**悲观估计**（$(1-\epsilon)$ 置信带的保守值 $t'(\epsilon), \gamma'(\epsilon)$）计算探索概率上界：

$$\hat{\tau} = \min_{\epsilon \in (0,1)} \frac{(1-\delta) - (1-\epsilon)\mathcal{L}(s(x), t'(\epsilon), \gamma'(\epsilon))}{1 - (1-\epsilon)\mathcal{L}(s(x), t'(\epsilon), \gamma'(\epsilon))} \ge \tau_{\text{nn}(x)}(s(x))$$

### 决策规则（Algorithm 2）

1. 计算嵌入，检索最近邻 $y = \text{nn}(x)$
2. 用 $\mathcal{O}(y)$ 拟合 sigmoid → $\hat{t}, \hat{\gamma}$
3. 遍历 $\epsilon \in (0,1)$ 计算 $\hat{\tau}$
4. 采样 $u \sim \text{Uniform}(0,1)$
5. 若 $u \le \hat{\tau}$ → explore（调用 LLM，更新 $\mathcal{O}(y)$）
6. 否则 → exploit（返回 $r(y)$）

**理论保证**（Theorem 4.1）：在 i.i.d. 和 sigmoid 正确建模假设下，对任意 prompt $x$ 和任意时刻 $n$：

$$\Pr(\mathbf{vCache}(x) = r(x) \mid \mathcal{D}) \ge 1 - \delta$$

## 实验关键数据

### 实验设置
- **嵌入模型**：GteLargeENv1-5、E5-large-v2、OpenAI text-embedding-3-small
- **LLM**：Llama-3-8B-Instruct、GPT-4o-mini
- **向量数据库**：HNSW + 余弦相似度
- **硬件**：Intel Xeon Platinum 8570 + NVIDIA Blackwell 192GB

### 基准数据集

| 基准 | 规模 | 特点 |
|------|------|------|
| SemCacheLMArena | 60K | LM-Arena 开放用户 prompt |
| SemCacheClassification | 45K | 3 个分类数据集 |
| SemCacheSearchQueries | 150K | Web 搜索查询 |
| SemCacheCombo | 27.5K | 混合，含无命中 prompt |

### 核心结果

**vs 静态阈值（GPTCache）**：
- SemCacheLMArena 上：最高 **26× 更低错误率**，**12.5× 更高命中率**
- SemCacheClassification 上：在 $\delta > 1.5\%$ 时全面优于静态基线
- $\delta < 1.5\%$ 时更保守（优先保证正确性，增加探索）

**错误率保证验证**（Figure 4）：
- vCache 在所有 $\delta$ 值下**始终满足错误率约束**
- GPTCache 错误率随样本增加**持续上升**，暴露静态阈值缺陷
- vCache 的缓存命中率持续增长（在线学习效应）

**Pareto 前沿对比**（Figure 5）：
- 在 error rate vs cache hit rate 的 Pareto 图中，vCache 在所有数据集上都支配（dominate）静态阈值方案
- 即使与微调嵌入的 GPTCache 对比，vCache 仍表现更优

### vs 微调嵌入基线
- vCache **无需训练**即可匹配/超越 Zhu et al. (2024) 的微调嵌入方法
- 微调嵌入在 OOD 数据上泛化差，vCache 的在线学习天然适应分布变化

## 亮点

1. **形式化正确性保证**：首次在语义缓存中提供用户可控的错误率上界 $\delta$，解决了生产级部署的信任问题
2. **逐嵌入自适应阈值**：Figure 3 展示了不同嵌入的最优阈值差异巨大，单一全局阈值不可行；vCache 自然捕捉这种变异
3. **零预训练在线学习**：模型无关、嵌入模型无关、无需标注数据，边推理边学习
4. **优雅的概率框架**：将 explore/exploit 决策形式化为在正确性概率约束下的最优化问题
5. **理论+实践统一**：既有 Theorem 4.1 的理论保证，又在大规模基准上验证

## 局限与展望

1. **长回复需 LLM-as-a-judge 判断等价**（Algorithm 1 L8），引入额外 LLM 调用（但可异步执行不影响延迟）
2. **依赖 i.i.d. 假设**：若 prompt 分布发生突变，保证可能暂时失效
3. **sigmoid 建模假设**：若真实正确性概率与相似度非 sigmoid 关系，分析可能不成立
4. **冷启动问题**：新嵌入无历史观测时，vCache 倾向保守（全部 explore）
5. **仅限语义相似场景**：对需要精确推理的 prompt（如数学计算），语义缓存本身不适用
6. **未评估多轮对话缓存**（论文聚焦单轮 prompt）

## 与相关工作的对比

| 方法 | 阈值类型 | 错误率保证 | 需要训练 | 模型无关 | 在线学习 |
|------|----------|:----------:|:--------:|:--------:|:--------:|
| GPTCache | 全局静态 | ✗ | ✗ | ✓ | ✗ |
| Azure/AWS Cache | 全局静态 | ✗ | ✗ | ✓ | ✗ |
| Zhu et al. 2024 | 全局静态 | ✗ | ✓ | ✗ | ✗ |
| SCalM | 全局静态 | ✗ | ✗ | ✓ | ✗ |
| EVM (Rudd et al.) | 逐类别 | ✗ | ✓ | — | ✗ |
| **vCache** | **逐嵌入动态** | **✓** | **✗** | **✓** | **✓** |

## 启发与关联

- **语义缓存 + RAG**：vCache 可直接集成到 RAG 系统中，缓存高频查询的检索结果
- **与 conformal prediction 的联系**：vCache 的概率保证框架与 conformal prediction 异曲同工，但无需标注校准集
- **在线学习范式**：将 explore/exploit 框架与正确性保证结合是一种可推广的设计模式（如推荐系统、A/B 测试）
- **多模态扩展**：语义缓存原理可扩展到图像/视频 prompt，vCache 的逐嵌入阈值机制天然适用

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 首个有正确性保证的语义缓存，逐嵌入在线阈值学习是全新方向
- 实验充分度: ⭐⭐⭐⭐ — 3 嵌入模型 × 2 LLM × 4 基准，但缺乏真实部署延迟对比
- 写作质量: ⭐⭐⭐⭐⭐ — 形式化严谨，motivation 清晰，图表设计出色（Figure 1-3 特别直观）
- 综合价值: ⭐⭐⭐⭐⭐ — 解决了语义缓存的核心信任问题，理论优雅且实用性极强，开源完善

<!-- RELATED:START -->

## 相关论文

- [Spectral Attention Steering for Prompt Highlighting](spectral_attention_steering_for_prompt_highlighting.md)
- [Prompt and Parameter Co-Optimization for Large Language Models](prompt_and_parameter_co-optimization_for_large_language_models.md)
- [Supercharging Floorplan Localization with Semantic Rays](../../ICCV2025/llm_evaluation/supercharging_floorplan_localization_with_semantic_rays.md)
- [Prompt and Parameter Co-Optimization for Large Language Model Task Adaptation](prompt_and_parameter_co-optimization_for_large_language_model_task_adaptation.md)
- [RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection](../../AAAI2026/llm_evaluation/refinevad_semantic-guided_feature_recalibration_for_weakly_supervised_video_anom.md)

<!-- RELATED:END -->
