---
title: >-
  [论文解读] Multilingual Encoder Knows More Than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages
description: >-
  [ACL2025][低资源语言] 提出 XLM-SWCM 框架，通过将多语言编码器权重复用到解码器中（CustomDecoderLayer 共享 + NormalDecoderLayer 随机初始化交替插入），以 457M 参数在极低资源语言（藏语）上超越 13B 参数的 MC2-LLaMA，藏语摘要 ROUGE-L 达 25.7 vs 16.1。
tags:
  - ACL2025
  - 低资源语言
  - 多语言模型
  - 权重共享
  - 编码器-解码器
  - 中国少数民族语言
---

# Multilingual Encoder Knows More Than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages

**会议**: ACL 2025  
**arXiv**: [2502.10852](https://arxiv.org/abs/2502.10852)  
**代码**: [GitHub](https://github.com/asd765973346/xlm-swcm)  
**领域**: LLM/NLP  
**关键词**: 多语言模型, 权重共享, 极低资源语言, 编码器-解码器, 藏语

## 一句话总结
提出 XLM-SWCM 框架，通过将多语言编码器权重复用到解码器中（CustomDecoderLayer 共享 + NormalDecoderLayer 随机初始化交替插入），以 457M 参数在极低资源语言（藏语）上超越 13B 参数的 MC2-LLaMA，藏语摘要 ROUGE-L 达 25.7 vs 16.1。

## 研究背景与动机
**领域现状**：大语言模型在高资源语言（英文、中文）上取得了显著进展，但对极低资源语言（如藏语 Tibetan、维吾尔语 Uyghur、蒙古语 Mongolian）的支持严重不足。这些语言在预训练语料中占比极低（通常 <0.01%），导致 LLM 在这些语言上的生成能力近乎为零。

**现有痛点**：(a) 直接在低资源语言上预训练 decoder-only LLM 面临数据量不足的根本问题——藏语维基百科仅约 1 万篇文章；(b) 现有多语言 LLM（如 BLOOM、LLaMA-2 多语言版）虽然号称支持多语言，但在极低资源语言上的实际生成质量极差；(c) 有趣的是，多语言编码器（如 XLM-R）在这些语言的理解任务（分类、NER）上表现尚可，但编码器模型天然不支持文本生成。

**核心矛盾**：多语言编码器（如 XLM-R）在极低资源语言上已学到可用的表示，但编码器架构不能直接做生成。同时从零训练 decoder 又缺少数据。如何利用编码器已学到的语言知识来初始化和加速 decoder 的学习？

**本文目标**：提出一种高效的权重复用策略，将多语言编码器的知识转移到编码器-解码器架构的 decoder 部分，使得小模型（457M）在极低资源语言上也能进行高质量的文本生成。

**切入角度**：观察到编码器和 decoder 的 Transformer 层结构有大量共享——self-attention 和 FFN 部分完全同构，区别仅在于 decoder 多了 cross-attention 层。因此可以将编码器权重直接复用到 decoder 的 self-attention 和 FFN，只需随机初始化 cross-attention。

**核心 idea**：通过 CustomDecoderLayer（共享编码器权重的 self-attention + FFN，随机初始化的 cross-attention）和 NormalDecoderLayer（完全随机初始化）的交替插入，以最优频率 X=3 组合，让编码器知识为 decoder 提供初始化——权重共享是最关键组件（去掉后性能下降 33%）。

## 方法详解

### 整体框架
XLM-SWCM (Cross-lingual Language Model with Shared Weight Cross-modal) 的核心思路：
1. **编码器**：直接使用预训练好的 XLM-R（frozen 或 partially frozen）
2. **解码器**：混合两种类型的层——
    - CustomDecoderLayer：self-attention 和 FFN 权重从 XLM-R 编码器复制，cross-attention 随机初始化
    - NormalDecoderLayer：所有参数随机初始化
3. **交替插入**：每 X 个 NormalDecoderLayer 插入 1 个 CustomDecoderLayer（X=3 为最优频率）
4. **训练**：在目标低资源语言的少量数据上进行生成任务训练（摘要、翻译等）

### 关键设计

1. **CustomDecoderLayer 设计**：

    - **共享部分**：将 XLM-R encoder layer 的 self-attention QKV 权重和 FFN 权重直接拷贝到 decoder layer 对应位置
    - **新增部分**：cross-attention 层（encoder-decoder attention）完全随机初始化
    - **设计动机**：编码器的 self-attention 已学会处理多语言 token 序列的注意力模式，FFN 已学会语言特定的特征变换——这些能力可直接迁移给 decoder，只需让 decoder 额外学会"如何关注编码器输出"（cross-attention 的功能）

2. **NormalDecoderLayer 设计**：

    - 完全随机初始化的标准 Transformer decoder 层
    - **存在意义**：纯共享权重会限制 decoder 学习生成特有模式的能力（如自回归 causal attention mask 导致的信息流差异）。交替插入随机层给模型提供了"自由度"来适应生成任务

3. **插入频率 X 的选择**：

    - X=1：每隔 1 个 Normal 插入 1 个 Custom（50% 共享）——共享比例最高
    - X=3：每隔 3 个 Normal 插入 1 个 Custom（25% 共享）——**最优**
    - X=6：每隔 6 个 Normal 插入 1 个 Custom（~14% 共享）——共享不足
    - **结论**：X=3 在"编码器知识利用"和"生成自由度"之间取得最佳平衡

4. **训练策略**：

    - 第一阶段：冻结编码器，只训练 decoder（含 Custom + Normal 层）
    - 第二阶段：解冻编码器低速学习率微调全模型
    - 数据增强：使用翻译对齐数据扩充藏语训练集（中→藏翻译对）

## 实验关键数据

### 主实验——藏语摘要

| 模型 | 参数量 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------|-------|---------|---------|---------|
| MC2-LLaMA | 13B | 22.3 | 8.9 | 16.1 |
| BLOOM | 7.1B | 19.8 | 7.2 | 14.3 |
| ChatGLM-3 | 6B | 20.5 | 7.8 | 15.2 |
| mBART-50 | 611M | 28.4 | 12.1 | 22.9 |
| XLM-R + random decoder | 457M | 24.1 | 9.5 | 19.2 |
| **XLM-SWCM (Ours)** | **457M** | **32.6** | **14.8** | **25.7** |

### 主实验——藏语翻译（藏→中）

| 模型 | 参数量 | BLEU | chrF |
|------|-------|------|------|
| MC2-LLaMA | 13B | 12.4 | 28.7 |
| BLOOM | 7.1B | 9.8 | 24.3 |
| mBART-50 | 611M | 18.2 | 35.6 |
| **XLM-SWCM** | **457M** | **21.5** | **39.8** |

### 消融实验

| 配置 | 藏语摘要 ROUGE-L | 相对下降 |
|------|-----------------|---------|
| XLM-SWCM (full, X=3) | 25.7 | — |
| 去掉权重共享（纯随机 decoder） | 17.2 | **-33%** |
| 去掉 NormalDecoder（纯共享） | 21.8 | -15% |
| X=1（50% 共享） | 23.1 | -10% |
| X=3（25% 共享，最优） | **25.7** | — |
| X=6（~14% 共享） | 22.4 | -13% |
| 不冻结编码器（直接全参数训练） | 23.9 | -7% |
| 不使用翻译数据增强 | 23.2 | -10% |

### 其他低资源语言验证

| 语言 | XLM-SWCM ROUGE-L | 最强基线 ROUGE-L | 相对提升 |
|------|------------------|----------------|---------|
| 藏语 (Tibetan) | 25.7 | 22.9 (mBART) | +12.2% |
| 维吾尔语 (Uyghur) | 23.4 | 20.1 (mBART) | +16.4% |
| 蒙古语 (Mongolian) | 27.1 | 24.6 (mBART) | +10.2% |

### 关键发现
1. **权重共享是最关键组件**：去掉后 ROUGE-L 从 25.7 暴跌到 17.2（-33%），这确认了编码器知识对 decoder 初始化的巨大价值
2. **小模型大胜大模型**：457M 的 XLM-SWCM 在藏语摘要上以 25.7 vs 16.1 ROUGE-L 碾压 13B 的 MC2-LLaMA——参数效率提升 28 倍
3. **X=3 是最优平衡点**：共享比例太高（X=1）限制生成自由度，太低（X=6）编码器知识利用不足
4. **两阶段训练优于直接训练**：先冻结编码器只训练 decoder，再解冻微调——相比直接全参数训练提升 7%
5. **跨语言迁移有效**：在藏语、维吾尔语、蒙古语三种极低资源语言上均超越所有基线，说明方法不是针对单一语言的

## 亮点与洞察
- **"编码器知识比你想象的多"**：标题本身就是最大洞察——XLM-R 这样的编码器虽然不能生成文本，但其内部表示已包含丰富的语言知识，这些知识可以通过权重复用高效迁移到 decoder
- **参数效率的极致**：用 3% 的参数量（457M vs 13B）实现 60% 的性能提升（25.7 vs 16.1），这对资源有限的低资源语言社区是巨大福音
- **CustomDecoder + NormalDecoder 交替设计优雅**：不是简单地全部共享或全部随机，而是通过频率控制在"知识保留"和"生成自由度"之间取得平衡，设计简洁但有效
- **对"编码器-解码器已过时"论调的反驳**：在 decoder-only 架构主导的时代，本文证明对于极低资源语言，编码器-解码器架构配合权重复用策略仍然是更优选择

## 局限与展望
- **仅测试摘要和翻译两种任务**：对话生成、问答、指令跟踪等其他生成任务未验证——权重共享策略在这些任务上是否同样有效？
- **编码器选择有限**：仅基于 XLM-R——如果换成更新更大的多语言编码器（如 XLM-R XXL 或 NLLB encoder），效果可能进一步提升
- **低资源语言范围仍有限**：测试了藏语/维吾尔语/蒙古语三种阿尔泰语系+藏语——非洲班图语系、南岛语系等其他低资源语言族尚需验证
- **与 LoRA 等参数高效方法的对比缺失**：如果对大 LLM 做目标语言的 LoRA 微调，是否能接近 XLM-SWCM？
- **X 的选择依赖实验搜索**：X=3 是实验找到的最优值，但缺少理论解释——为什么 25% 的共享比例是最佳？

## 相关工作与启发
- **与 mBART/mT5 的关系**：mBART 也用编码器-解码器架构处理多语言，但未做"编码器→解码器权重共享"——XLM-SWCM 在此基础上更进一步
- **与知识蒸馏的关系**：可以将权重初始化视为一种"硬蒸馏"——不通过 logit/representation matching 传递知识，而是直接复制权重
- **对 LLM 多语言扩展的启发**：当需要为 LLM 添加新的极低资源语言支持时，与其从零训练不如利用已有多语言模型的权重——"知识复用"比"知识重建"更高效
- **实用场景**：藏语、维吾尔语等少数民族语言的 NLP 工具（自动摘要、机器翻译）对文化保护和信息可及性有重要社会价值

## 评分
- 新颖性: ⭐⭐⭐⭐ CustomDecoder + NormalDecoder 交替设计是有创意的架构贡献，"编码器→解码器权重复用"的思路直觉简单但之前较少被系统研究
- 实验充分度: ⭐⭐⭐⭐ 多语言验证 × 全面消融 × 参数量天差地别的基线对比，实验设计有说服力
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，消融实验逻辑清楚，motivation 阐述到位
- 价值: ⭐⭐⭐⭐⭐ 对极低资源语言 NLP 有直接实用价值，457M 模型可在消费级 GPU 上运行，对低资源社区友好

**会议**: ACL2025  
**arXiv**: [2502.10852](https://arxiv.org/abs/2502.10852)  
**代码**: [GitHub](https://github.com/asd765973346/xlm-swcm)  
**领域**: llm_nlp  
**关键词**: 低资源语言, 多语言模型, 权重共享, 编码器-解码器, 中国少数民族语言

## 一句话总结
