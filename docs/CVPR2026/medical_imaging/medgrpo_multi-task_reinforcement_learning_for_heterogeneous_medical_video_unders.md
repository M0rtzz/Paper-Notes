---
title: >-
  [论文解读] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding
description: >-
  [CVPR 2026][医学图像][医学视频理解] MedGRPO 提出了两项关键创新解决医学视频多数据集强化学习中的训练崩溃问题：跨数据集奖励归一化（用 logistic 函数将不同难度数据集的中位表现映射到相同奖励值）和医学 LLM 评审（通过五个临床维度的比较性评分），基于 Qwen2.5-VL-7B 在 MedVidBench（532K 视频指令对）上超越 GPT-4.1 和 Gemini-2.5-Flash。
tags:
  - CVPR 2026
  - 医学图像
  - 医学视频理解
  - 强化学习
  - 跨数据集奖励归一化
  - VLM微调
  - 多任务学习
---

# MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding

**会议**: CVPR 2026  
**arXiv**: [2512.06581](https://arxiv.org/abs/2512.06581)  
**代码**: https://uii-america.github.io/MedGRPO/  
**领域**: 医学图像 / 视频理解  
**关键词**: 医学视频理解、强化学习、跨数据集奖励归一化、VLM微调、多任务学习

## 一句话总结

MedGRPO 提出了两项关键创新解决医学视频多数据集强化学习中的训练崩溃问题：跨数据集奖励归一化（用 logistic 函数将不同难度数据集的中位表现映射到相同奖励值）和医学 LLM 评审（通过五个临床维度的比较性评分），基于 Qwen2.5-VL-7B 在 MedVidBench（532K 视频指令对）上超越 GPT-4.1 和 Gemini-2.5-Flash。

## 研究背景与动机

1. **领域现状**：大型视觉-语言模型在通用视频理解上取得了显著进展，但在医学视频理解上表现大幅退化。医学视频理解需要精细的手术动作解读、领域特定术语（如区分"grasper"和"tool"）、手术安全评估和多阶段时序推理。

2. **现有痛点**：
    - **缺乏指令跟随格式的训练数据**：现有医学视频数据集（CholecT50、EgoSurgery 等）有丰富标注但非 QA 对话格式
    - **标准 RL 在异质数据集上训练崩溃**：不同数据集难度差异极大（如 CoPESD 时空定位中位 mIoU~0.5 vs EgoSurgery~0.12），标准 GRPO 的原始奖励导致模型过拟合简单数据集、放弃困难数据集
    - **通用语义相似度度量无法捕获临床差异**："The tool grasps tissue" vs "The grasper dissects the cystic duct" 余弦相似度 ≈0.82 但临床含义完全不同

3. **核心矛盾**：如何在难度差异极大的异质医学视频数据集上进行平衡的多任务强化学习？

4. **切入角度**：中位公平性（median fairness）——中位表现在所有数据集-任务对上获得相同的归一化奖励，消除梯度更新中的偏差。

5. **核心 idea**：用 logistic 奖励归一化实现跨数据集公平优化，用医学 LLM 评审替代通用语义相似度捕获临床细粒度。

## 方法详解

### 整体框架

两阶段训练范式：
1. **SFT 阶段**：在 MedVidBench 上对 Qwen2.5-VL-7B 做监督微调，注入领域知识并建立基线性能和百分位统计
2. **GRPO 阶段**：使用 MedGRPO 进行强化学习——采样 8 组响应，计算跨数据集归一化奖励，用 GRPO 的组内优势估计更新策略

输入：医学视频帧（自适应采样 0.1-3 FPS）+ 指令 → 输出：跨 8 种任务的文本/定位结果

### 关键设计

1. **MedVidBench 数据构建与质量保证管线**:

    - 功能：将现有专家标注系统化转换为大规模指令跟随 QA 对
    - 核心思路：三阶段管线——(1) 专家标注提示：按数据源特化策略，对帧标注数据集（CholecT50 等）在帧上叠加边界框和标签；对网络来源数据集（AVOS 等）用 Whisper-X 提取音频转录并加入视频元数据。(2) 双模型生成：GPT-4.1 和 Gemini-2.5-Flash 独立生成描述，防止单模型偏差。(3) 质量验证：计算两个模型输出的句子相似度，过滤低质量对（相似度<0.3），按源视频划分训练/测试（比例 0.85/0.15）。最终 532K 样本，覆盖 8 个数据源 × 8 种任务（视频级/段级/帧级）。
    - 设计动机：医学标注转换为 QA 格式需要专家级理解，人工标注成本高；利用 VLM 转换+双模型验证是可扩展的高质量方案。

2. **跨数据集奖励归一化**:

    - 功能：让不同难度的数据集-任务对在优化中获得公平的梯度贡献
    - 核心思路：对每个数据集-任务对 $(d,t)$，应用 logistic 变换：$r_{norm}^{(d,t)}(x) = \frac{1}{1 + \exp(-k \cdot \frac{x - p_{50}^{(d,t)}}{IQR^{(d,t)}})}$，其中 $p_{50}$ 是中位数，$IQR = p_{75} - p_{25}$ 是四分位距，$k=3.0$ 控制斜率。百分位统计从 SFT 基线预测计算。关键性质：当 $x = p_{50}$ 时归一化奖励恒为 0.5（中位公平性），logistic 函数提供处处非零梯度（无死区），IQR 缩放对异常值鲁棒。
    - 设计动机：未归一化时训练直接崩溃——CVS 从 0.894 跌至 0.020，STG 从 0.177 跌至 0.010，TAG 从 0.142 跌至 0.004。训练 entropy 变得高度不稳定。原因是简单数据集的高幅值奖励主导梯度更新。

3. **医学 LLM 评审（Medical LLM Judge）**:

    - 功能：评估医学描述的细粒度临床正确性
    - 核心思路：使用 GPT-4.1 做比较性相似度评分（"生成描述与参考有多接近？"而非绝对质量评级），避免分数膨胀。评估五个临床维度（1-5分）：医学术语精度、器械解剖识别、具体性vs模糊性、手术流程上下文、动作准确性。混合设计：最终奖励 = 50% 归一化语义相似度 + 50% 归一化 LLM 评审分，兼顾段落级语义一致性和细节级临床正确性。
    - 设计动机：标准嵌入度量无法区分 "tool" vs "grasper"、"grasps" vs "dissects"、"tissue" vs "cystic duct" 等临床关键差异。比较性评分比绝对评分更好地区分模型质量差异。

### 损失函数 / 训练策略

GRPO 目标函数使用非对称裁剪（$\epsilon_{low}=0.2$, $\epsilon_{high}=0.3$），允许更大正更新同时约束负更新。移除标准 GRPO 的 KL 惩罚项。SFT 训练 3 epochs, LR $5 \times 10^{-7}$；GRPO 训练 5000 步, LR $5 \times 10^{-7}$，组大小 $G=8$。定位任务使用乘法复合奖励（格式惩罚）。所有实验在 8×H100 GPU 上运行。

## 实验关键数据

### 主实验

| 模型 | CVS acc | STG mIoU | TAG@0.3 | TAG@0.5 | VS llm | RC llm |
|------|---------|----------|---------|---------|--------|--------|
| GPT-4.1 | 0.018 | 0.014 | 0.096 | 0.005 | 2.490 | 2.080 |
| Gemini-2.5-Flash | 0.101 | 0.047 | 0.045 | 0.021 | 2.352 | 1.912 |
| Qwen2.5VL-7B (off-shelf) | 0.105 | 0.020 | 0.006 | 0.068 | 2.452 | 2.090 |
| Qwen2.5VL-7B SFT | 0.894 | 0.177 | 0.142 | 0.091 | 3.596 | 2.757 |
| **Qwen2.5VL-7B MedGRPO** | **0.896** | **0.202** | **0.216** | **0.156** | **4.184** | **3.442** |

### 消融实验

| 配置 | CVS | STG | TAG@0.3 | VS llm | RC llm |
|------|-----|-----|---------|--------|--------|
| A: 完整 MedGRPO | 0.896 | 0.202 | 0.216 | 4.184 | 3.442 |
| B: 无奖励归一化 | 0.020 | 0.010 | 0.004 | 1.061 | 3.469 |
| C: 仅 TAG+STG | 0.914 | 0.193 | 0.202 | 3.776 | 3.425 |
| D: VS+RC 有 LLM judge | 0.894 | 0.183 | 0.149 | 3.824 | 3.235 |
| E: VS+RC 无 LLM judge | 0.894 | 0.183 | 0.140 | 3.733 | 2.984 |

### 关键发现

- **奖励归一化是生死攸关的**：去掉归一化后所有指标崩溃（Row B），CVS 从 0.896 跌至 0.020，证明这不是锦上添花而是必要条件
- **多任务协同显著**：加入描述任务（VS+RC）的奖励反而提升了定位任务：STG +4.7%、TAG@0.3 +6.9%（Row A vs C）
- **LLM 评审贡献明确**：有 LLM judge 的 VS 比无 LLM judge 高 0.091（3.824 vs 3.733），RC 高 0.251（3.235 vs 2.984）
- **SFT 已远超闭源模型**：Qwen2.5VL-7B SFT 的 CVS 0.894 vs GPT-4.1 的 0.018，差距达 50 倍
- **跨模型泛化**：同样的管线应用到 Qwen3-VL-4B 也获得一致提升（STG +0.043，TAG@0.3 +0.039）
- **2026 模型仍不够**：GPT-5.4 在 STG 上仅 0.004，说明医学视频理解仍需领域适配

## 亮点与洞察

- **logistic 奖励归一化的通用性**：中位公平性原则可以直接应用到任何多数据集/多任务 RL 训练中（不限于医学）。关键设计是用 IQR 而非 min-max 缩放，对异常值更鲁棒。这个技巧对多任务 RLHF 有直接参考价值。
- **比较性评分替代绝对评分**：LLM 评审采用"与参考有多接近"而非"绝对质量多高"的提问方式，避免了分数膨胀问题。这种评估策略值得在其他 LLM-as-judge 场景中推广。
- **数据驱动的领域适配依然是王道**：即使是 GPT-5.4，在医学视频定位上仍然几乎为零。领域特定数据+微调的路线在医学领域依然不可替代。

## 局限与展望

- **LLM 评审成本高**：每个训练样本需要调用 GPT-4.1 评估，限制了训练规模和速度
- **百分位统计的静态性**：归一化所用的 $p_{25}, p_{50}, p_{75}$ 来自 SFT 基线，但 RL 训练过程中分布在变化，可能需要动态更新
- **仅对 4 种任务做 GRPO**：CVS、NAP、SA 等准确率任务未直接纳入 RL 训练
- **GRPO 训练仅 5000 步**：RL 阶段训练量相对 SFT 较少，可能未充分收敛
- **双模型验证的覆盖偏差**：GPT-4.1 和 Gemini-2.5-Flash 的共同盲区可能被遗漏
- **未探索开源 LLM 做 judge**：对 GPT-4.1 的依赖增加了成本和不可控性

## 相关工作与启发

- **vs SurgLLM/SurgLaVi**：这些方法仅在单个手术数据集上训练，缺乏跨手术类型的泛化；MedVidBench 覆盖 8 个数据源实现了跨领域训练
- **vs VideoChat-R1.5**：通用视频 RL 模型在医学任务上完全失败（CVS=0.000），说明医学 RL 需要领域特化的奖励设计
- **vs DAPO**：MedGRPO 借鉴了 DAPO 的非对称裁剪和去KL惩罚，但增加了跨数据集归一化这一关键组件

## 评分

- 新颖性: ⭐⭐⭐⭐ 跨数据集奖励归一化和医学LLM评审的组合设计有实用创新性，但各组件技术门槛不高
- 实验充分度: ⭐⭐⭐⭐⭐ 8个任务、多个基线（含2026最新模型）、跨模型验证、详尽消融
- 写作质量: ⭐⭐⭐⭐ 技术描述清晰，问题动机阐述充分，定性分析直观
- 价值: ⭐⭐⭐⭐⭐ 建立了医学视频理解的基础设施（数据集+训练范式+评估方法），对领域有长远影响

<!-- RELATED:START -->

## 相关论文

- [OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)
- [Boosting Medical Visual Understanding From Multi-Granular Language Learning](../../ICLR2026/medical_imaging/boosting_medical_visual_understanding_from_multi-granular_language_learning.md)
- [Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)
- [OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation](orapo_oracle-educated_reinforcement_learning_for_data-efficient_and_factual_radi.md)

<!-- RELATED:END -->
