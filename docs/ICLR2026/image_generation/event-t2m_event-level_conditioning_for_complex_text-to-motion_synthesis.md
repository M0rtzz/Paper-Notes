---
title: >-
  [论文解读] Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis
description: >-
  [ICLR 2026][图像生成][文本到动作生成] 提出 Event-T2M 框架，将文本提示分解为事件级别的原子动作，结合 TMR 编码器和事件级交叉注意力（ECA）模块注入 Conformer 扩散模型，显著提升多事件复杂动作生成的质量和语义对齐。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "文本到动作生成"
  - "事件级条件"
  - "扩散模型"
  - "组合动作"
  - "Conformer"
---

# Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis

**会议**: ICLR 2026  
**arXiv**: [2602.04292](https://arxiv.org/abs/2602.04292)  
**代码**: 有 (项目页面)  
**领域**: 图像生成  
**关键词**: 文本到动作生成, 事件级条件, 扩散模型, 组合动作, Conformer

## 一句话总结

提出 Event-T2M 框架，将文本提示分解为事件级别的原子动作，结合 TMR 编码器和事件级交叉注意力（ECA）模块注入 Conformer 扩散模型，显著提升多事件复杂动作生成的质量和语义对齐。

## 研究背景与动机

文本到动作（Text-to-Motion）生成领域虽然在 HumanML3D 和 KIT-ML 等基准上取得了显著进展（FID 已优化到小数点后两位），但这些基准主要由简单单动作描述组成，掩盖了一个关键问题：当面对复杂多动作提示（如"向前跑，然后停下，然后挥手"）时，现有系统往往会合并、跳过或重排动作。

核心问题在于：

**现有方法将整个提示压缩为单一嵌入**：大多数方法使用 CLIP 的 [EOS] token 作为全局表示，丢失了时序信息

**基准不区分简单与复杂提示**：无法评估模型在组合复杂度增加时的表现

**CLIP 预训练于图文对**：对动作的时序连续性和事件转换缺乏监督信号

## 方法详解

### 整体框架

Event-T2M 的核心思路是将文本到动作生成重新定义为事件级别的条件生成问题，包含三个关键组件：

1. **LLM 事件分解**：利用 Gemini 2.5 Flash 将输入文本提示 $W$ 分割为事件序列 $\{C_k\}_{k=1}^K$
2. **TMR 事件编码**：用运动感知的 TMR 编码器将每个事件映射为事件 token
3. **ECA 注入**：通过事件级交叉注意力模块在 Conformer 块中融合事件信息

**事件的形式化定义**：事件是文本提示中最小的语义自包含动作或状态变化，其执行可以在时间上被隔离并映射到连续运动片段。例如，"A person steps backward, jumps up, runs forward, then runs backward" 被分解为四个事件。

### 关键设计

#### 1. 事件 Token 生成

每个事件 $C_k$ 通过 TMR 编码器编码为事件 token：

$$E_k = f_{\text{TMR}}(C_k), \quad E_k \in \mathbb{R}^{D_y}$$

堆叠所有事件 token 得到 $E \in \mathbb{R}^{K \times D_y}$。同时引入全局文本 token $G = f_{\text{TMR}}(W)$ 作为整体语义补充，在局部事件线索模糊时提供全局语义后备。

#### 2. Event-T2M Block 架构

模型堆叠 $N$ 个相同块，每个块包含 8 步更新：

| 步骤 | 模块 | 功能 |
|------|------|------|
| (1) | LIMM | 局部信息建模（深度可分离卷积） |
| (2) | ATII | 自适应文本信息注入（通道级门控） |
| (3) | FFN | 前馈网络（0.5 残差权重） |
| (4) | ConformerSA | 自注意力（全局时序依赖） |
| (5) | **ECA** | **事件级交叉注意力（核心）** |
| (6) | ConformerConv | 深度可分离卷积（局部动力学） |
| (7) | FFN | 前馈网络（0.5 残差权重） |
| (8) | LIMM | 局部信息建模 |

#### 3. 事件级交叉注意力（ECA）

ECA 是核心创新，将 Conformer 块中的标准自注意力替换为运动到文本的交叉注意力机制：

- **Query**：来自运动 token $x_t^{\text{ctx}}$
- **Key/Value**：来自事件 token $E$

$$Q_m = x_t^{\text{ctx}} W^Q, \quad K_e = E W^K, \quad V_e = E W^V$$

$$A^{(h)} = \text{softmax}\left(\frac{Q_m^{(h)} (K_e^{(h)})^\top}{\sqrt{d_h}}\right)$$

使用可学习缩放因子 $\gamma$（初始化为接近零）以确保训练稳定：$\text{ECA}(x_t, E) = \gamma \cdot \text{Dropout}(Z)$

#### 4. ATII 自适应文本注入

ATII 通过通道级门控将全局文本嵌入 $G$ 与局部运动状态融合：

$$\hat{g}_j = \text{Sigmoid}(W_c[m'_j \oplus G]) \odot G$$

先对运动序列做 $S$ 倍下采样，再通过门控机制自适应过滤全局语义。

### 损失函数 / 训练策略

采用标准条件去噪扩散目标函数，训练去噪器 $\varphi_\theta$ 从噪声运动 $x_t$ 恢复干净运动 $x_0$：

$$\mathcal{L}(\theta) = \mathbb{E}_{x_0, t, \epsilon}\left[\|x_0 - \varphi_\theta(x_t, t, G, E)\|_2^2\right]$$

- 训练时以概率 $\tau$ 随机丢弃文本条件，实现 Classifier-Free Guidance (CFG)
- 推理时采用 10 步 DDPM 进行高效生成
- 0.5 残差权重用于 FFN，遵循 Macaron 风格架构的直觉

## 实验关键数据

### 主实验

**表1：HumanML3D 标准基准**

| 方法 | R-Prec Top-1↑ | R-Prec Top-3↑ | FID↓ | MM-Dist↓ |
|------|--------------|--------------|------|----------|
| MoMask | 0.521 | 0.807 | 0.045 | 2.958 |
| MoGenTS | 0.529 | 0.812 | 0.033 | 2.867 |
| **Event-T2M** | **0.562** | **0.842** | 0.056 | **2.711** |

**表3：HumanML3D-E 事件分层基准（≥4 事件）**

| 方法 | R-Prec Top-1↑ | FID↓ | MM-Dist↓ |
|------|--------------|------|----------|
| MoMask | 0.441 | 0.418 | 3.205 |
| MoGenTS | 0.420 | 0.423 | 3.241 |
| **Event-T2M** | **0.466** | 0.265 | **3.063** |

Event-T2M 在 R-Precision Top-1 上高出 MoGenTS 约 4.6 个百分点（≥4 事件），展示出在复杂组合场景中的优势。

### 消融实验

**文本编码器对比（TMR vs CLIP）**：事件级条件下，TMR 编码器在所有事件复杂度上均优于 CLIP。

**条件方式对比**：事件级条件（Event-level）vs 逐 token 条件（Token-level）：

| 条件方式 | R-Prec Top-1↑ (≥2事件) | FID↓ |
|---------|----------------------|------|
| Token-level | 0.521 | 0.082 |
| **Event-level** | **0.536** | 0.079 |

事件级编码在所有复杂度条件下均优于逐 token 编码。

### 关键发现

1. **事件复杂度增加时优势放大**：随着事件数从 ≥1 到 ≥4 增加，基线方法性能急剧下降，而 Event-T2M 保持稳健
2. **效率优势**：在 ≥4 事件条件下，Event-T2M 以较小的模型规模实现高精度
3. **人类评估验证**：事件定义的合理性、HumanML3D-E 的可靠性以及生成质量均获得人类评估者的高度认可

## 亮点与洞察

1. **事件的形式化定义**具有普适性——将复杂提示分解为最小语义自包含单元的思路可推广到其他条件生成任务
2. **TMR 替代 CLIP**：用运动语言对齐的 TMR 编码器替代通用 CLIP，为特定领域的条件生成提供了范式参考
3. **HumanML3D-E 基准**：首个按事件数量分层的评估基准，填补了组合复杂度评估的空白
4. **可学习缩放因子 $\gamma$**：在 ECA 中初始化接近零确保训练稳定性，是一个实用的工程技巧

## 局限与展望

1. LLM 事件分解依赖外部模型（Gemini 2.5 Flash），增加了推理依赖和延迟
2. 事件之间的过渡质量（transition quality）未被显式建模
3. 仅在 HumanML3D/KIT-ML 上验证，缺少更大规模数据集上的泛化实验
4. 事件数量增大时 FID 仍有一定上升空间
5. 可探索端到端的事件分解与生成联合优化

## 相关工作与启发

- **GraphMotion**：用语义图增强文本表示，但评估有限
- **AttT2M**：body-part 注意力 + 全局-局部运动文本注意力
- **MMM**：掩码运动建模，联合编码文本和运动
- **Light-T2M**：ATII 模块的灵感来源
- 启发：事件级分解的思路可迁移到文本到视频、文本到舞蹈等任务

## 评分

- 新颖性：⭐⭐⭐⭐ — 事件级条件化是一个简洁且有效的新视角
- 技术贡献：⭐⭐⭐⭐ — ECA + TMR + 事件分层基准三位一体
- 实验充分度：⭐⭐⭐⭐ — 标准基准 + 分层基准 + 消融 + 人类评估
- 写作质量：⭐⭐⭐⭐ — 结构清晰，动机明确
- 总体推荐：⭐⭐⭐⭐ — 值得关注的工作，尤其对多动作生成场景有实际价值

## 背景与动机

## 核心问题

## 方法详解

## 实验关键数据

## 亮点

## 局限与展望

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] ANCHOR: LLM-driven Subject Conditioning for Text-to-Image Synthesis](../../ACL2026/image_generation/anchor_llm-driven_subject_conditioning_for_text-to-image_synthesis.md)
- [\[ICLR 2026\] Localized Concept Erasure in Text-to-Image Diffusion Models via High-Level Representation Misdirection](localized_concept_erasure_in_text-to-image_diffusion_models_via_high-level_repre.md)
- [\[ICCV 2025\] Efficient Input-Level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation](../../ICCV2025/image_generation/efficient_input-level_backdoor_defense_on_text-to-image_synthesis_via_neuron_act.md)
- [\[ICML 2025\] How to Move Your Dragon: Text-to-Motion Synthesis for Large-Vocabulary Objects](../../ICML2025/image_generation/how_to_move_your_dragon_text-to-motion_synthesis_for_large-vocabulary_objects.md)
- [\[ICML 2025\] Broadband Ground Motion Synthesis by Diffusion Model with Minimal Condition](../../ICML2025/image_generation/broadband_ground_motion_synthesis_by_diffusion_model_with_minimal_condition.md)

</div>

<!-- RELATED:END -->
