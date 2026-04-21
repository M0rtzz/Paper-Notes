---
title: >-
  [论文解读] ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering
description: >-
  [CVPR 2026][KB-VQA] 提出 ReAG，一个推理增强的多模态 RAG 方法，结合粗细粒度检索与 Critic 过滤模型减少噪声，并通过 GRPO 强化学习训练生成器进行显式推理，在知识密集型 VQA 上达到新 SOTA。
tags:
  - CVPR 2026
  - KB-VQA
  - RAG
  - 强化学习
  - 推理增强
  - 多模态检索
---

# ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering

**会议**: CVPR 2026  
**arXiv**: [2511.22715](https://arxiv.org/abs/2511.22715)  
**代码**: [aimagelab.github.io/ReAG](https://aimagelab.github.io/ReAG)  
**领域**: 知识增强视觉问答 / 检索增强生成  
**关键词**: KB-VQA, RAG, 强化学习, 推理增强, 多模态检索

## 一句话总结

提出 ReAG，一个推理增强的多模态 RAG 方法，结合粗细粒度检索与 Critic 过滤模型减少噪声，并通过 GRPO 强化学习训练生成器进行显式推理，在知识密集型 VQA 上达到新 SOTA。

## 研究背景与动机

知识密集型视觉问答（KB-VQA）要求模型回答超出视觉内容本身的领域特定问题，需要从外部知识库（如 Wikipedia）中检索相关信息来辅助回答。即便是最先进的多模态大语言模型（MLLMs），在面对预训练数据中表示不足的领域知识时也表现不佳。

现有的检索增强方法存在两个核心问题：

**检索噪声大**：用户查询高度异质，外部知识库可达百万级文档，导致检索召回率低、噪声多，向 MLLM 输入了大量无关段落

**推理能力弱**：即使检索到了相关文档，从中提取正确信息并推理出答案并非易事，现有方法缺乏对检索内容的显式推理能力

ReAG 的核心思路是**先过滤、再推理**：通过多层次检索 + Critic 过滤减少噪声输入，再通过强化学习训练生成器具备对检索内容的显式推理能力。

## 方法详解

### 整体框架

ReAG 包含四个主要阶段：
1. 多层次检索阶段（粗粒度 + 细粒度）
2. Critic 模型过滤阶段
3. 生成器冷启动 SFT 阶段
4. 强化学习训练阶段

### 关键设计

1. **多层次检索（Multi-level Retrieval）**：

    - **粗粒度检索**：使用 EVA-CLIP-8B 将整个查询图像编码，通过余弦相似度从知识库中检索 top-k 文档，获取候选段落集 $\mathcal{P}^{cg}$
    - **细粒度检索**：利用 GroundingDINO 检测问题中提到的视觉主体，裁剪出关注区域后再次检索，获取 $\mathcal{P}^{fg}$，弥补全图检索遗漏的细节
    - 两阶段结果按相关性合并排序，保留 top-k 文档的所有段落形成 $\mathcal{P}^{noisy}$
    - 设计动机：单一粗粒度检索召回率不足，细粒度检索聚焦于问题相关的视觉区域可提高召回

2. **Critic 过滤模型**：

    - 基于 Qwen2.5-VL-3B 微调的自回归 MLLM，接收 $(I_q, q, p)$ 输入，判断段落 $p$ 是否与问题相关
    - 仅保留预测"Yes"概率高于阈值的段落，形成 $\mathcal{P}^{relevant}$
    - 设计动机：增加 k 值虽然提高召回但降低精度，Critic 模型可有效剔除噪声段落，且与检索骨干解耦，可适配任意检索引擎

3. **生成器冷启动（Cold Start SFT）**：

    - 采用 DeepSeek-R1 启发的多阶段训练策略，先进行 SFT 建立初始推理能力
    - 从 MLLM 收集高质量推理轨迹 $tr$，以 `<think>` 和 `<answer>` 特殊标记分隔推理过程与最终答案
    - 损失函数：$\mathcal{L}_{SFT} = \alpha \mathcal{L}_A + (1-\alpha)\mathcal{L}_T$，其中 $\alpha=0.8$ 给予答案更大权重

4. **GRPO 强化学习训练**：

    - 基于 GRPO 框架，融合 DAPO 的改进（去除 KL 散度惩罚、token 级损失计算）
    - 每次迭代用 $(I_q, q, p)$ 生成 N=8 个补全，通过规则奖励计算优势值
    - 设计动机：SFT 只是冷启动，RL 进一步提升模型对检索证据的推理质量和鲁棒性

### 损失函数 / 训练策略

- **奖励设计**：$R_i = \gamma R_{task}(o_i) + \delta R_{fmt}(o_i)$，其中 $\gamma=1.0$, $\delta=0.2$
    - 任务奖励：按问题类型（数值/文本、单答案/多答案）解析并验证正确性
    - 格式奖励：检查是否遵循 `<think>...<answer>...` 模板
- 冻结视觉编码器，仅更新 MLP 适配器和 LLM 权重
- RL 阶段使用 Adam 优化器，学习率 $1 \times 10^{-6}$，每批 128 个提示、每提示 8 个补全

## 实验关键数据

### 主实验

使用 EVA-CLIP-8B 作为检索器的结果：

| 数据集 | 指标 | ReAG (3B) | ReflectiVA (3B) | 提升 |
|--------|------|-----------|-----------------|------|
| E-VQA (All) | Accuracy | 42.9 | 35.2 | +7.7 |
| InfoSeek (All) | Accuracy | 43.3 | 38.9 | +4.3 |

| 数据集 | 指标 | ReAG (7B) | VLM-PRF (InternVL3-8B) | 提升 |
|--------|------|-----------|------------------------|------|
| E-VQA (Single-Hop) | Accuracy | 44.9 | 40.1 | +4.8 |
| E-VQA (All) | Accuracy | 47.0 | 39.2 | +7.8 |
| InfoSeek (All) | Accuracy | 47.2 | 42.5 | +4.7 |

使用 OMGM 检索器时性能进一步提升：ReAG (7B) 在 E-VQA 上达到 52.5%，InfoSeek 上达到 49.2%。

使用 Oracle Wikipedia 页面（上界实验）：ReAG (7B) 在 E-VQA 上达到 81.5%，InfoSeek 上达到 59.7%。

### 消融实验

| 配置 | E-VQA (Single-Hop) | InfoSeek (All) | 说明 |
|------|-------|---------|------|
| 无检索（零样本） | 21.9 | 18.3 | 仅靠内部知识 |
| 粗粒度检索 | 19.2 | 10.1 | 噪声段落反而降低性能 |
| 粗+细粒度+Critic | 40.2 | 27.1 | 过滤显著提升 |
| +SFT | 39.3 | 37.5 | 推理能力大幅提升InfoSeek |
| +SFT+推理轨迹 | 38.1 | 41.3 | 显式推理进一步提升 |
| +SFT+RL (ReAG完整) | 41.3 | 43.3 | RL 带来最终提升 |

### 关键发现

1. **Critic 过滤至关重要**：无 Critic 时，粗粒度检索的噪声段落甚至会让性能低于零样本，说明噪声管理是 RAG 系统的关键
2. **RL 优于纯 SFT**：强化学习阶段在两个基准上均带来显著提升，验证了基于奖励的推理优化的有效性
3. **推理轨迹具有可解释性**：模型生成的推理过程可以揭示检索段落的有用性和推导步骤，提供完全的可解释性
4. **ReAG 对检索骨干是无关的（agnostic）**：Critic 模型可无缝置于任何检索引擎之上

## 亮点与洞察

- **过滤优先的设计理念**：不同于大多数 RAG 方法试图让生成器学会处理噪声，ReAG 先通过 Critic 从源头减少噪声，再让生成器专注于高质量推理
- **SFT → RL 的多阶段训练策略**：借鉴 DeepSeek-R1 的思路，SFT 仅作为冷启动建立初始推理行为，RL 负责真正提升推理质量
- **细粒度检索的互补性**：通过检测问题中的视觉主体并裁剪图像，获取与问题更相关的检索结果，与粗粒度检索形成有效互补
- **定量验证了 RAG 中噪声问题的严重性**：消融实验清晰展示未过滤的检索结果甚至会降低性能

## 局限与展望

1. Critic 模型本身可能存在误判，更精细的相关性评估（如分段质量评分而非二分类）可能更好
2. 检索阶段使用固定 top-k，自适应检索数量可能进一步提升效率
3. 当前仅在 Wikipedia 知识库上验证，对其他领域（如医学、法律）的泛化能力未知
4. 推理轨迹的生成增加了推理时间，实际部署时需要平衡推理深度与延迟

## 相关工作与启发

- **ReflectiVA**：使用控制 token 指导检索和知识评估，但缺乏显式推理
- **VLM-PRF**：利用外部工具进行知识过滤，与 ReAG 的 Critic 思想类似但实现不同
- **DeepSeek-R1 / GRPO**：提供了 RL 增强推理的方法论基础
- **Search-R1**：将检索与推理集成用于复杂查询，为 ReAG 的多模态扩展提供启发

## 评分

- 新颖性: ⭐⭐⭐⭐ （Critic过滤+RL推理的组合有效但各组件并非全新）
- 实验充分度: ⭐⭐⭐⭐⭐ （两个标准基准、多种检索器、详细消融、oracle上界）
- 写作质量: ⭐⭐⭐⭐⭐ （结构清晰，消融全面，图表直观）
- 价值: ⭐⭐⭐⭐⭐ （为知识增强VQA提供了完整且高效的解决方案）

<!-- RELATED:START -->

## 相关论文

- [Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering](../../NeurIPS2025/reinforcement_learning/knowledge-based_visual_question_answer_with_multimodal_processing_retrieval_and_.md)
- [TAdaRAG: Task Adaptive Retrieval-Augmented Generation via On-the-Fly Knowledge Graph Construction](../../AAAI2026/reinforcement_learning/tadarag_task_adaptive_retrieval-augmented_generation_via_on-the-fly_knowledge_gr.md)
- [Seeing is Improving: Visual Feedback for Iterative Text Layout Refinement](seeing_is_improving_visual_feedback_for_iterative_text_layout_refinement.md)
- [AnyDoc: Enhancing Document Generation via Large-Scale HTML/CSS Data Synthesis and Height-Aware Reinforcement Optimization](anydoc_enhancing_document_generation_via_large-scale_htmlcss_data_synthesis_and_.md)
- [BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)

<!-- RELATED:END -->
