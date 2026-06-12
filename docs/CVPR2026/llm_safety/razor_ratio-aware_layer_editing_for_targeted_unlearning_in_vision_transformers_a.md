---
title: >-
  [论文解读] RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models
description: >-
  [CVPR2026][LLM安全][machine unlearning] 提出 RAZOR，一种基于比率感知的多层/多头选择性编辑框架，可在 CLIP、Stable Diffusion 和 VLM 等 Transformer 视觉模型中高效精准地完成目标遗忘，同时保持模型整体性能与量化鲁棒性。
tags:
  - "CVPR2026"
  - "LLM安全"
  - "machine unlearning"
  - "Transformer"
  - "扩散模型"
  - "CLIP"
  - "model editing"
  - "ratio-aware saliency"
---

# RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models

**会议**: CVPR2026  
**arXiv**: [2603.14819](https://arxiv.org/abs/2603.14819)  
**代码**: [raviranjan-ai/RAZOR-cvpr2026](https://github.com/raviranjan-ai/RAZOR-cvpr2026)  
**领域**: 图像生成  
**关键词**: machine unlearning, vision transformer, diffusion model, CLIP, model editing, ratio-aware saliency

## 一句话总结

提出 RAZOR，一种基于比率感知的多层/多头选择性编辑框架，可在 CLIP、Stable Diffusion 和 VLM 等 Transformer 视觉模型中高效精准地完成目标遗忘，同时保持模型整体性能与量化鲁棒性。

## 研究背景与动机

大规模视觉-语言模型（CLIP）、文本生成图像扩散模型（Stable Diffusion）和视觉-语言助手（LLaVA）在海量数据上训练后会嵌入敏感或不期望的信息（如人脸身份、版权内容），出于 GDPR 等合规要求需要将这些知识从模型中移除。然而从头重训成本极高，因此 **机器遗忘（machine unlearning）** 成为重要研究方向。

理想的遗忘方法需要同时满足三个标准：

**高效**：遗忘计算开销低；

**精准**：仅移除目标信息，不影响无关能力；

**鲁棒**：遗忘后即使在对抗/量化条件下目标知识也不可恢复。

现有方法各有局限：
- **SSD** 使用 Fisher 信息选择参数，计算开销大且效用损失明显；
- **SalUn** 基于梯度显著性筛选权重，但仅考虑 forget 梯度，导致遗忘不彻底；
- **SLUG** 首创单层编辑思路，效率极高，但当知识分布在多层时单层编辑过于脆弱；
- **ESD/FMN/UCE** 等扩散模型方法在遗忘完整性和内容保真度之间仍存在权衡。

核心问题在于：现有方法的参数选择**仅由 forget 集显著性驱动**，retain 冲突只在事后被缓解，无法避免 forget/retain 动态耦合。RAZOR 的出发点正是用 **比率感知评分** 在参数选择阶段就联合考虑遗忘压力和保留对齐。

## 方法详解

### 整体框架

RAZOR（Ratio-Aware Zero/One-step Optimized Retentive unlearning）想解决的是：怎么把敏感知识从视觉 Transformer 里精准抹掉，又不伤到无关能力、还扛得住量化。它是个模型无关的轻量框架，整条流水线分三步走——先用比率感知显著性给每一层打分，筛出真正该编辑的层集合 $\mathcal{K}$；再在这些层上做一次约束多目标的融合梯度更新，把遗忘、保留、稳定三个目标一起优化；如果一轮下来遗忘还不够，就迭代地把新层加进 $\mathcal{K}$ 继续编辑，直到达标或触顶。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["预训练模型 θ + 遗忘集 D_f + 保留集 D_r"] --> B["比率感知显著性评分<br/>逐层算一步 forget/retain 梯度，按 φ(l) 打分"]
    B -->|"φ(l) > τ 入选"| C["待编辑层集合 𝒦"]
    C --> D["三部分损失约束更新<br/>在 𝒦 上联合优化 保留+遗忘+稳定"]
    D --> E{"遗忘达标？"}
    E -->|"未达标 且 t ≤ 6"| F["迭代扩展 𝒦<br/>重算 φ_t(l)，并入得分最高的新层"]
    F --> D
    E -->|"达标"| G["输出遗忘后模型"]
```

### 关键设计

**1. 比率感知显著性评分：在选层阶段就同时看「能不能忘」和「会不会误伤」**

以往方法的通病是参数选择只被 forget 集驱动——SLUG 只挑单层、SalUn 只看 forget 梯度，retain 冲突只能事后再补，于是 forget 和 retain 动态耦合、顾此失彼。RAZOR 把这个判断提前到选层阶段。对每一层 $l$ 先算一步 forget/retain 梯度 $g^f_l = \nabla_{\theta_l} \mathcal{L}_{\text{forget}}$、$g^r_l = \nabla_{\theta_l} \mathcal{L}_{\text{retain}}$，再算比率感知显著性：

$$\phi(l) = \frac{\|g^f_l\|_2}{\|\theta_l\|_2 + \varepsilon} \cdot (1 - \cos(g^f_l, g^r_l))^\alpha$$

第一项是归一化后的 forget 梯度范数，衡量这一层对遗忘贡献多大；第二项衡量 forget 与 retain 梯度有多正交——越正交，说明编辑这层对保留知识的附带损伤越小。$\alpha \in [0,1]$ 在量级和正交性之间调权重，$\varepsilon$ 保数值稳定，最后按阈值筛 $\mathcal{K} = \{l \mid \phi(l) > \tau\}$。一次梯度就同时评估了「遗忘能力」和「保留安全」，这是全文最核心的设计。

**2. 三部分损失：让保效用、做遗忘、防漂移各管一摊**

选好层之后，更新方向由一个三项损失主导：

$$\mathcal{L}_{\text{RAZOR}} = \mathcal{L}_{\text{retain}} + \lambda_f \rho \, \mathcal{L}_{\text{forget}} + \lambda_m \, \mathcal{L}_{\text{mismatch}}$$

三项各司其职：$\mathcal{L}_{\text{retain}}$ 保住无关能力，$\mathcal{L}_{\text{forget}}$ 用梯度上升把目标知识推开，$\mathcal{L}_{\text{mismatch}}$ 是稳定正则、避免嵌入空间发生全局漂移；$\rho \in (0,1]$ 是比率超参，控制遗忘压力的整体强度。同一套框架在三类模型上只需替换每一项的实例：

| 损失项 | CLIP | Stable Diffusion | VLM (LLaVA) |
|---|---|---|---|
| $\mathcal{L}_{\text{retain}}$（保持效用） | 对称 InfoNCE 对比损失 | ε-prediction 去噪损失 | 对称 InfoNCE（视觉编码器） |
| $\mathcal{L}_{\text{forget}}$（推开遗忘） | 余弦嵌入推远损失 | 文本编码器上的 CE 损失 | 视觉编码器上的 CE 损失 |
| $\mathcal{L}_{\text{mismatch}}$（稳定正则） | 相似度漂移正则(SDR) | 生成结果的 SDR | 中性 QA 上的 logit 漂移正则 |

**3. 迭代扩展层集合：一轮编辑不够，就按需加层直到遗忘达标**

比率感知评分一次只筛出当前看来最该编辑的一批高分层，但目标知识可能比预想的更分散——初次编辑后遗忘未必充分。RAZOR 不靠一次性大改解决，而是渐进扩展：每轮 $t$ 在更新后的模型上**重新**计算显著性 $\phi_t(l)$（注意是用 $\theta^{(t)}$ 重算，因为编辑过的模型上各层的贡献会变），把当前得分最高的那一层并入 $\mathcal{K}$ 再做一次融合更新，直到遗忘指标越过预设门槛或触及迭代上限（$t \le 6$）。这种「按需加层」既保证遗忘门槛可达，又避免一次过度编辑伤及保留知识——它正是 RAZOR 相对单层方法（SLUG）和固定多层方法的核心区别：编辑范围由遗忘是否达标动态决定，而非事先固定。

### 损失函数 / 训练策略

选好层、定下三项损失后，对每个选中层 $l \in \mathcal{K}$ 执行一步融合梯度更新：

$$\Delta\theta_l = -\eta_l(-\lambda_f \rho \, g^f_l + g^r_l + \lambda_m \nabla_{\theta_l}\mathcal{L}_{\text{mismatch}})$$

括号里就是三部分损失各自梯度的线性组合——遗忘项取负号做梯度上升、保留项与稳定项做下降。每层步长 $\eta_l$ 用一次轻量二分搜索确定：在小验证集上挑出能给出最佳遗忘-保留权衡的最大稳定步长，既不过冲也不过保守。这套「先算梯度、再一步更新」的设计沿用了单层方法的高效性，又推广到多层/多头编辑；若一轮后遗忘仍不达标，则交给上面的迭代扩展继续加层。

## 实验关键数据

### 主实验 1：CLIP 身份遗忘（Table 3 精选）

| 方法 | CIFAR-10 M1↓ | CIFAR-10 M4↑ | CIFAR-10 M5↑ | ImageNet M1↓ | ImageNet M4↑ | LAION M1↓ | LAION M4↑ |
|---|---|---|---|---|---|---|---|
| SSD | 52.00 | 25.00 | 97.50 | 52.50 | 30.00 | 42.00 | 48.00 |
| SalUn | 97.00 | 83.00 | 84.50 | 88.00 | 84.00 | 48.00 | 88.00 |
| SLUG | 67.50 | 87.50 | 96.50 | 68.00 | 88.00 | 48.00 | 88.00 |
| **RAZOR** | **52.50** | **89.00** | **100.0** | **53.50** | **92.00** | **40.00** | **94.00** |

RAZOR 在遗忘精度（M1 低）的同时实现了最高的 retain 精度（M4）和完美的检索稳定性（M5=100），且 M3 隐私泄露为 0。量化到 4-bit 后 RAZOR 的性能衰减最小（M5 仅降 1.4%），远优于 SalUn 和 LoTUS。

### 主实验 2：Stable Diffusion 风格/物体遗忘（Table 4 精选）

| 方法 | SD-V3 Style UA↑ | SD-V3 Style IRA↑ | SD-V3 Object UA↑ | SD-V1.5 Style UA↑ | SD-V1.5 Object UA↑ |
|---|---|---|---|---|---|
| ESD | 99.62 | 89.97 | 97.44 | 98.58 | 92.15 |
| SalUn | 90.36 | 92.33 | 91.06 | 86.26 | 86.91 |
| SLUG | 88.20 | 85.59 | 85.44 | 86.29 | 75.43 |
| **RAZOR** | **99.40** | **98.97** | **98.80** | **99.26** | **97.91** |

RAZOR 在 SD-V3 和 SD-V1.5 上的 UA/IRA/CRA 三项指标全面领先，特别是 CRA 几乎达到 100，表明遗忘后的语义一致性极佳。

### 效率对比（Table 5，SD-V1.5）

| 方法 | 时间(s)↓ | 显存(GB)↓ | 存储(GB)↓ | Trade-off↑ |
|---|---|---|---|---|
| ESD | 6163 | 17.8 | 4.30 | 11.97 |
| SLUG | 39 | 3.6 | 0.04 | 59.42 |
| **RAZOR** | 78 | 4.2 | 0.06 | **66.86** |

RAZOR 虽然比 SLUG 慢一倍，但综合遗忘精度后的性能-效率权衡得分最高（66.86 vs 59.42），且只需修改少量层权重（存储仅 0.06 GB）。

### 消融实验（Table 7，CLIP on CIFAR-10）

| 配置 | M1↓ | M3→0 | M4↑ | M5↑ |
|---|---|---|---|---|
| 无 Retain loss | 52.72 | 0.40 | 82.00 | 99.0 |
| 无 Mismatch loss | 53.25 | 1.04 | 88.00 | 100.0 |
| 无 Forget loss | 96.00 | 0.40 | 86.00 | 99.0 |
| 全层更新（无选择） | 51.00 | 1.58 | 78.00 | 96.0 |
| 有选择无迭代 | 53.00 | 0.00 | 88.82 | 100.0 |
| **完整 RAZOR** | **52.50** | **0.00** | **89.00** | **100.0** |

关键发现：
- 去掉 forget loss 遗忘基本失败（M1=96%）；
- 全层更新虽然 M1 最低但 M3 泄露最高、M4 效用最差——验证了选择性编辑的必要性；
- 迭代扩展带来 M4 从 88.82→89.00 的微幅提升，更重要的是确保遗忘门槛可达。

### VLM 遗忘（Table 6，LLaVA-1.6-8B）

对 10 个名人身份遗忘后，平均 FA 降至仅 2.2%（基线 97.25%），同时 GQA 保持 60.46（基线 60.18）、MMBench 保持 60.9%（基线 61.57%），通用多模态能力几乎未受损。

## 亮点与洞察

1. **比率感知选择**是一个简洁有效的设计：一次梯度计算就能同时评估 forget 贡献与 retain 安全性，比先 forget 再补 retain 的顺序策略更稳定。
2. **模型无关性**：同一套框架统一适用于 CLIP、SD、LLaVA 三类典型视觉 Transformer，仅需替换损失函数实例化。
3. **量化鲁棒性**是一个被忽视但实际部署中至关重要的维度——RAZOR 在 4-bit 量化后仍保持接近原精度的遗忘效果，而 SalUn、LoTUS 等全模型更新方法在量化后性能显著退化。
4. 效率-精度权衡得分（Trade-off=66.86）超过所有基线，说明多层编辑带来的额外计算成本被更好的遗忘效果完全补偿。

## 局限与展望

1. 迭代扩展上限固定为 6 轮，没有讨论更复杂任务是否需要更多迭代，也没有自适应停止准则；
2. 阈值 $\tau$ 和超参 $\alpha, \rho, \lambda_f, \lambda_m$ 的敏感性分析仅在附录中，主文没有给出调参通用指南；
3. 仅在身份/风格/物体三类遗忘场景评估，缺少对更复杂组合概念（如"特定人穿特定衣服"）的测试；
4. 未讨论对抗性恢复攻击（adversarial unlearning）下的鲁棒性；
5. 尚未扩展到音频、视频等模态。

## 相关工作与启发

- **SLUG** (ICML 2025)：RAZOR 的直接基线，单层编辑效率高但脆弱，RAZOR 通过多层选择性编辑解决了知识分布式存储的问题。
- **SalUn** (NeurIPS 2023)：仅用 forget 梯度做显著性排序，RAZOR 加入 retain 梯度的正交性指标形成比率感知。
- **ESD**：负引导采样方式做概念压制，在 SD-V3 上 UA 略高但 IRA/CRA 远低于 RAZOR。
- **启发**：比率感知的「forget-retain 正交性」思路可推广到 LLM 知识编辑、持续学习中的遗忘-保留权衡。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 比率感知显著性评分和迭代扩展思路有创新，但整体仍是梯度-编辑范式的工程改进
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三类模型(CLIP/SD/VLM)×多数据集×量化鲁棒性×消融×效率对比，非常全面
- **写作质量**: ⭐⭐⭐⭐ — 框架清晰、损失表格对比直观，数学表述规范
- **价值**: ⭐⭐⭐⭐ — 为部署级视觉模型遗忘提供了实用且可扩展的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Targeted Unlearning with Single Layer Unlearning Gradient](../../ICML2025/llm_safety/targeted_unlearning_with_single_layer_unlearning_gradient.md)
- [\[AAAI 2026\] Attention Retention for Continual Learning with Vision Transformers](../../AAAI2026/llm_safety/attention_retention_for_continual_learning_with_vision_transformers.md)
- [\[CVPR 2026\] DAMP: Class Unlearning via Depth-Aware Removal of Forget-Specific Directions](damp_class_unlearning_via_depth_aware_removal_of_forget_specific_directions.md)
- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [\[CVPR 2025\] Hyperbolic Safety-Aware Vision-Language Models](../../CVPR2025/llm_safety/hyperbolic_safety-aware_vision-language_models.md)

</div>

<!-- RELATED:END -->
