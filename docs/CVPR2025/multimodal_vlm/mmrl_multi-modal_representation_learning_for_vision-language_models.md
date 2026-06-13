---
title: >-
  [论文解读] MMRL: Multi-Modal Representation Learning for Vision-Language Models
description: >-
  [CVPR 2025][多模态VLM][CLIP迁移学习] MMRL 提出了一个共享的、模态无关的可学习表征空间，将表征 token 投影到图像和文本编码器的高层（保留低层泛化知识），并通过解耦推理策略（基类用表征+类别特征，新类只用类别特征）在 15 个数据集上实现了 few-shot 适配与泛化的最优平衡，刷新了 base-to-novel 泛化的 SOTA。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "CLIP迁移学习"
  - "多模态表征"
  - "少样本学习"
  - "提示学习"
  - "泛化性保持"
---

# MMRL: Multi-Modal Representation Learning for Vision-Language Models

**会议**: CVPR 2025  
**arXiv**: [2503.08497](https://arxiv.org/abs/2503.08497)  
**代码**: [https://github.com/yunncheng/MMRL](https://github.com/yunncheng/MMRL)  
**领域**: 多模态VLM  
**关键词**: CLIP迁移学习, 多模态表征, 少样本学习, prompt learning, 泛化性保持

## 一句话总结
MMRL 提出了一个共享的、模态无关的可学习表征空间，将表征 token 投影到图像和文本编码器的高层（保留低层泛化知识），并通过解耦推理策略（基类用表征+类别特征，新类只用类别特征）在 15 个数据集上实现了 few-shot 适配与泛化的最优平衡，刷新了 base-to-novel 泛化的 SOTA。

## 研究背景与动机
CLIP 等预训练 VLM 具有强大的零样本能力，但在 few-shot 下游适配时经常过拟合，导致对新类/新数据集的泛化能力下降。现有方法分两大流派：(1) Prompt Learning（如 CoOp、MaPLe）通过可学习 prompt 适配，但在浅层注入 prompt 会破坏 CLIP 的泛化知识，且以文本为中心的设计导致模态不平衡；(2) Adapter（如 CLIP-Adapter、MMA）通过轻量模块调整特征，但只优化 class token 特征，在数据稀少时容易过拟合。核心矛盾是：**如何在适配下游任务的同时保持 VLM 的泛化能力？** MMRL 的切入角度是引入一个独立的、无偏的共享表征空间，在高层编码器中进行多模态交互，让 representation token 学习下游知识而 class token 保持泛化。

## 方法详解

### 整体框架
MMRL 基于冻结的 CLIP 模型，引入一个共享可学习表征空间 $\mathcal{R}$（高斯初始化），通过可学习映射函数 $\mathcal{F}$ 将空间 token $R \in \mathbb{R}^{K \times d_r}$ 投影为视觉表征 token $R^v$ 和文本表征 token $R^t$，从第 $J$ 层开始插入到图像和文本编码器的 Transformer 层中，与原始 token 一起参与注意力计算。整个 CLIP 模型保持冻结，只训练 $\mathcal{R}$、$\mathcal{F}$ 和表征 token 的投影层。

### 关键设计
1. **共享模态无关表征空间**:
    - 功能：建立一个不偏向任何模态的桥梁，促进平衡的多模态交互
    - 核心思路：用高斯分布初始化共享空间 $\mathcal{R}$，通过独立的线性映射 $\mathcal{F}_i^v$ 和 $\mathcal{F}_i^t$ 分别为每层生成视觉和文本表征 token。关键在于**每层独立映射**——不同层的表征 token 由不同映射函数生成，适应不同层的特征分布
    - 设计动机：MaPLe 等方法从文本 prompt 映射视觉 prompt，本质上以文本为中心，忽略了视觉模态的独立性。共享空间的设计让双模态从同一起点出发，实现真正的"无偏"多模态学习

2. **高层注入策略**:
    - 功能：保护 CLIP 低层的泛化知识不被破坏
    - 核心思路：表征 token 只在第 $J$ 层及之后注入。对于图像编码器，$J$ 层之前正常处理 $[c_{i-1}, E_{i-1}]$，之后处理 $[c_{i-1}, R_{i-1}^v, E_{i-1}]$，但表征 token 的输出在中间层被丢弃（只在最后一层 $L$ 保留）。文本编码器中，$R^t$ 插入到文本 token 之前，并调整注意力掩码适配增加的序列长度
    - 设计动机：MMA 发现 CLIP 编码器高层编码数据集相关的判别性特征，低层编码泛化特征。在低层注入可学习参数会搅乱泛化表征，导致新类性能骤降

3. **解耦推理策略 (Decoupled Inference)**:
    - 功能：在基类和新类上使用不同的特征组合，最大化各自性能
    - 核心思路：图像编码器最后一层输出 class token 特征 $f_c = P_v^c(c_L)$（冻结投影）和表征特征 $f_r = P_v^r(\text{Mean}(R_L^v))$（可学习投影）。基类推理时两者加权融合 $p(y=c|x) = \alpha \cdot p(y=c|f_c) + (1-\alpha) \cdot p(y=c|f_r)$；新类推理时只用 $f_c$（保留泛化知识）
    - 设计动机：消融实验证明（w/o DS2），如果新类也用双特征，表征 token 过拟合基类后会拖累新类性能。分离使用是关键

### 损失函数 / 训练策略
总损失为：$\mathcal{L}_{MMRL} = \alpha \mathcal{L}_{ce}^c + (1-\alpha) \mathcal{L}_{ce}^r + \lambda (\mathcal{L}_{cos}^v + \mathcal{L}_{cos}^t)$

- $\mathcal{L}_{ce}^c$ 和 $\mathcal{L}_{ce}^r$：分别对 class token 特征和 representation token 特征的交叉熵损失
- $\mathcal{L}_{cos}^v$ 和 $\mathcal{L}_{cos}^t$：正则项，约束 class token 的图像/文本特征与冻结 CLIP 的零样本特征保持余弦相似度，防止适配过程中偏离预训练知识
- $\alpha = 0.7$ 控制双特征权重，$\lambda = 0.5$ 控制正则强度

## 实验关键数据

### 主实验（Base-to-Novel 泛化，11 数据集平均）

| 方法 | Base | Novel | HM | 说明 |
|------|------|-------|-----|------|
| CLIP | 69.34 | 74.22 | 71.70 | 零样本基线 |
| CoOp | 82.69 | 63.22 | 71.66 | 基类强但新类暴跌 |
| MaPLe | 82.28 | 75.14 | 78.55 | 多模态 prompt |
| PromptSRC | 84.26 | 76.10 | 79.97 | 自正则 prompt |
| MMA (prev SOTA) | 83.20 | 76.80 | 79.87 | 多模态 adapter |
| **MMRL** | **85.68** | **77.16** | **81.20** | Base/Novel/HM 全面领先 |

### 消融实验

| 配置 | Base | Novel | HM | 说明 |
|------|------|-------|-----|------|
| w/o V (去掉视觉表征) | 82.83 | 75.03 | 78.74 | 视觉表征贡献很大 |
| w/o L (去掉文本表征) | 85.05 | 75.65 | 80.08 | 文本表征也重要 |
| w/o DS₁ (新类也用双特征) | 83.59 | 77.16 | 80.25 | 解耦策略关键 |
| w/o DS₂ (基类只用类特征) | 85.68 | 73.80 | 79.30 | 表征特征对基类提升大 |
| w/o RS (无共享空间) | 85.79 | 75.55 | 80.34 | 共享空间对新类泛化重要 |
| MMRL† (有偏映射如MaPLe) | 85.60 | 76.02 | 80.55 | 无偏 > 有偏 |
| **MMRL (完整)** | **85.68** | **77.16** | **81.20** | 所有组件协同 |

### 关键发现
- MMRL 在 11 数据集上 HM 平均提升 MMA 1.33%，其中 Base 提升 2.48% 而 Novel 也保持提升 0.36%
- 表征空间维度 $d_r = 512$ 最优，过大导致过拟合
- 表征 token 应在较高层注入（第 9 层最优，共 12 层编码器），太低会破坏泛化特征
- $K = 4$ 个表征 token 是最优平衡点

## 亮点与洞察
- **解耦推理**是最关键的设计——训练时同时优化两种特征，推理时按需选择，巧妙地解决了适配 vs 泛化的矛盾
- 共享模态无关空间的设计优于从一个模态映射到另一个模态的有偏方案（消融实验 MMRL† vs MMRL 验证）
- 整个方法只需训练少量参数（$\mathcal{R}$、$\mathcal{F}$、一个投影层），CLIP 完全冻结，训练高效
- 在 FGVCAircraft 等细粒度数据集上，MMRL Base 准确率比 PromptSRC 高 3.57%
- 表征 token 在中间层的输出被丢弃、只在最后一层保留，避免了中间层信息泄漏
- 正则项 $\mathcal{L}_{cos}$ 同时约束视觉和文本侧的 class feature，从两端防止特征漂移
- 在 few-shot 学习中，随着 shot 数增加优势更明显，说明适配能力上限更高

## 局限与展望
- 方法专门针对 CLIP 类双编码器结构设计，不直接适用于生成式 VLM（如 LLaVA）
- 16-shot 以下的极端少样本场景提升有限
- 正则项需要额外的冻结 CLIP 前向传播来计算零样本特征，增加训练开销
- 超参数 $\alpha$、$\lambda$、$J$、$K$、$d_r$ 较多，调参成本不低

## 相关工作与启发
- **vs MaPLe**: MaPLe 在浅层注入 prompt 且以文本为中心映射视觉 prompt；MMRL 在高层注入且使用无偏共享空间
- **vs MMA**: MMA 是多模态 adapter 但只优化 class token；MMRL 引入独立的 representation token 分工
- **vs PromptSRC**: PromptSRC 用自正则防止遗忘；MMRL 用解耦策略 + 余弦正则，效果更好
- **vs CoOp/CoCoOp**: CoOp 完全不考虑泛化导致新类暴跌，CoCoOp 通过 instance-specific prompt 部分缓解
- **vs ProVP**: ProVP 仅用单模态视觉 prompt，缺乏跨模态交互

## 补充说明
- 基于 ViT-B/16 CLIP 模型，编码器共 12 层，表征 token 从第 9 层开始注入
- 训练使用 SGD 优化器，16-shot 设置下仅需少量迭代即可收敛

## 评分
- 新颖性: ⭐⭐⭐⭐ 共享表征空间 + 解耦推理组合有创意，但整体仍在 prompt/adapter 范式框架内
- 实验充分度: ⭐⭐⭐⭐⭐ 15 数据集、4 种评测设置、详尽消融（6 个维度）
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，方法描述详尽
- 价值: ⭐⭐⭐⭐ 在 CLIP 高效迁移这一拥挤赛道上仍取得了稳定的 SOTA 提升

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[CVPR 2025\] GeoMM: On Geodesic Perspective for Multi-Modal Learning](geomm_on_geodesic_perspective_for_multi-modal_learning.md)
- [\[CVPR 2025\] Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector](rethinking_vision-language_model_in_face_forensics_multi-modal_interpretable_for.md)
- [\[CVPR 2025\] NLPrompt: Noise-Label Prompt Learning for Vision-Language Models](nlprompt_noise-label_prompt_learning_for_vision-language_models.md)
- [\[CVPR 2025\] EgoLM: Multi-Modal Language Model of Egocentric Motions](egolm_multi-modal_language_model_of_egocentric_motions.md)

</div>

<!-- RELATED:END -->
