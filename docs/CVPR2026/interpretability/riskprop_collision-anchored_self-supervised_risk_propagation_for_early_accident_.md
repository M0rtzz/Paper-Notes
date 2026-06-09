---
title: >-
  [论文解读] RiskProp: Collision-Anchored Self-Supervised Risk Propagation for Early Accident Anticipation
description: >-
  [CVPR 2026][可解释性][事故预测] 提出 RiskProp，一种以碰撞帧为锚点的自监督风险传播范式，通过未来帧正则化损失和自适应单调约束损失，仅依赖碰撞帧标注即可学习时序连贯的风险演化曲线，在 CAP 和 Nexar 数据集上达到 SOTA。
tags:
  - "CVPR 2026"
  - "可解释性"
  - "事故预测"
  - "自监督风险传播"
  - "时序建模"
  - "单调性约束"
  - "行车记录仪"
---

# RiskProp: Collision-Anchored Self-Supervised Risk Propagation for Early Accident Anticipation

**会议**: CVPR 2026  
**arXiv**: [2603.27165](https://arxiv.org/abs/2603.27165)  
**代码**: [https://github.com/xingyueye5/RiskProp/](https://github.com/xingyueye5/RiskProp/)  
**领域**: 可解释性  
**关键词**: 事故预测, 自监督风险传播, 时序建模, 单调性约束, 行车记录仪

## 一句话总结

提出 RiskProp，一种以碰撞帧为锚点的自监督风险传播范式，通过未来帧正则化损失和自适应单调约束损失，仅依赖碰撞帧标注即可学习时序连贯的风险演化曲线，在 CAP 和 Nexar 数据集上达到 SOTA。

## 研究背景与动机

1. **领域现状**：事故预期（Accident Anticipation）旨在通过行车记录仪视频实时估计风险分数，当分数超过阈值时触发预警。现有方法将该任务建模为二分类监督学习——无事故视频所有帧标为 0，事故视频从"异常起始帧"到碰撞帧标为 1。

2. **现有痛点**：二元标签范式存在根本缺陷——它强迫模型将所有碰撞前帧视为等同风险，忽略了风险的渐进演化特性。手动标注"异常起始"帧主观且不一致，不同标注者之间差异大，导致噪声监督信号。

3. **核心矛盾**：真实驾驶中风险是连续递增的过程（如驾驶员分心时缓慢上升，行人突然出现时急剧飙升），但二元标签无法表达这种中间状态和场景依赖的风险演化。

4. **本文目标** 不依赖手动标注的异常起始帧，仅利用可靠标注的碰撞帧，学习时序连贯且物理合理的风险演化曲线。

5. **切入角度**：作者提出两个关键观察——(1) 未来帧包含更明确的碰撞证据，模型对未来帧的预测更准确，可作为当前帧的伪监督；(2) 碰撞前风险总体呈非递减趋势。

6. **核心 idea**：以碰撞帧为唯一锚点，通过下一帧预测值作为软标签反向传播风险信号，结合自适应单调约束实现无需人工标注的风险演化建模。

## 方法详解

### 整体框架

RiskProp 想解决的核心问题是：在只有碰撞帧这一可靠标注的前提下，学出一条时序连贯、随风险渐增的风险曲线，而不是被二元标签强行拉平。整条 pipeline 很轻——对每个时间步 $t$，模型吃进连续 $O$ 帧（实验中 $O=5$）$\mathbf{x}_t = \{x_{t-O+1}, \dots, x_t\}$，先用 3D CNN 编码器（SlowOnly）抽特征，再过 sigmoid 输出该步风险分数 $a_t = \sigma(f_\theta(\mathbf{x}_t))$。真正的巧思全在监督信号的设计上：唯一锚点是碰撞帧的真标签，中间帧不靠人工标，而是让"未来帧的预测"反过来给"当前帧"当软目标，再叠一层随场景松紧的单调约束。三个损失分工明确——BCE 只钉住碰撞帧和起始帧两端，未来帧正则化（FFR）负责把风险信号从碰撞帧往前传，自适应单调约束（AMC）负责让传出来的曲线大体不回头。

### 关键设计

**1. 未来帧正则化损失（FFR）：用"下一帧"给"当前帧"当软标签，把风险从碰撞帧反向灌回早期帧**

二元标签的根本缺陷在于把所有碰撞前帧当成同等风险，而中间帧又没法可靠标注；FFR 的切入点是一个朴素事实——越接近碰撞，证据越充分，模型对未来帧的判断天然比当前帧更可信。于是它用 stop-gradient 把下一帧的预测 $\text{detach}(z_{t+1})$ 冻成当前帧 $z_t$ 的目标，最小化 $\mathcal{L}_{\text{reg}} = \sum_{t=1}^{T-1} \|\text{detach}(z_{t+1}) - z_t\|^2$。因为碰撞帧 $z_T$ 有唯一可靠的真标签 $y_T=1$ 钉住，这个高风险信号会沿 $z_T \to z_{T-1} \to \cdots$ 一层层链式回传到早期帧。stop-gradient 是关键：它保证信息只往前（时间上往早）单向流动，不会让早期帧的不确定预测反噬污染碰撞帧这个唯一可信的锚，整套机制因此不需要额外的 teacher 模型，也彻底绕开了主观的"异常起始"标注。

**2. 自适应单调约束损失（AMC）：让风险曲线总体非递减，但容忍短期波动，松紧随场景自适应**

光有 FFR 还不够——风险信号回传后仍可能出现局部抖动或倒挂，而真实驾驶中危险度是总体递增的过程。AMC 对随机采样的帧对 $(i,j)$（$j>i$）施加 $a_j \geq a_i$，写成 hinge 形式 $\mathcal{L}_{\text{mono}} = \frac{1}{|\mathcal{D}|} \sum_{(i,j)} \max\big(0,\, a_i - a_j + \delta(\Delta t, \bar{c}_{i:j})\big)$。这里的关键不是硬约束，而是那个会自适应的容忍边距

$$\delta = \delta_0 \cdot \Delta t \cdot \bar{c}_{i:j},$$

它同时随帧对时间距离 $\Delta t$ 和区间平均预测置信度 $\bar{c}_{i:j}$ 缩放：时间跨度越大、越确信处于高风险段，边距越大、单调约束越严；反之在置信度低或相邻帧之间则放松，给短期波动留出余地。固定边距或硬单调会把曲线过度压成阶梯，自适应边距则在"曲线该涨"和"别把噪声也强行单调化"之间取得平衡。

**3. 仅碰撞帧标注策略：只标客观可靠的碰撞时刻，中间全交给自监督**

现有方法要么靠指数衰减加权、要么靠人工标注"异常起始"窗口来定义正样本，而这个起始点本身主观、标注者间差异大，是噪声监督的源头。RiskProp 把标注砍到极简——事故视频里只有碰撞帧记正（$y_T=1$）、起始帧记负（$y_0=0$），所有中间帧的监督完全由 FFR 软标签提供；非事故视频则全帧标 0。BCE 在这两端施加时对碰撞帧加更高权重，以缓解正负样本不平衡。之所以敢这么省，是因为碰撞时间戳是客观可观测的事实，而 FFR+AMC 已经把中间段的演化补齐；消融也证实这套极简标注能逼平用人工起始帧的密集标注。

### 损失函数 / 训练策略

总损失为 $\mathcal{L} = \mathcal{L}_{\text{bce}} + \lambda_1 \cdot \mathcal{L}_{\text{reg}} + \lambda_2 \cdot \mathcal{L}_{\text{mono}}$，取 $\lambda_1=1.5$、$\lambda_2=1.1$。训练用 SlowOnly 预训练权重、SGD 优化器，8 卡 A800 跑 50 epochs，batch size 64，初始学习率 0.002 并每 20 epochs 衰减至 10%；帧采样区间 $d_{\min}=0.1$、$d_{\max}=0.9$，单调边距基数 $\delta_0=0.01$；视频帧统一 resize 到 224×224，帧率重采样为 10 FPS。

## 实验关键数据

### 主实验

| 数据集 | 方法 | mAUC0.1 | mAUC | mAP | mTTA0.1 (s) |
|--------|------|---------|------|-----|-------------|
| CAP | AdaLEA | 0.379 | 0.807 | 0.857 | 1.115 |
| CAP | CRASH | 0.401 | 0.842 | 0.887 | 1.085 |
| CAP | **RiskProp** | **0.483** | 0.853 | **0.890** | **1.207** |
| Nexar | CRASH | 0.393 | 0.832 | 0.846 | 0.857 |
| Nexar | **RiskProp** | **0.472** | **0.869** | **0.870** | **0.958** |

在 Nexar 上，RiskProp 在所有指标上均超越第二名 CRASH，mAUC0.1 提升 0.079，mAUC 提升 0.037，mAP 提升 0.024。

### 消融实验

| 配置 | 标注策略 | mAUC0.1 (CAP) | mAUC0.1 (Nexar) | 说明 |
|------|---------|---------------|-----------------|------|
| Baseline (无 FFR/AMC) | Only Collision | 0.358 | 0.298 | 仅碰撞帧标注，无自监督约束 |
| +FFR | Only Collision | 0.474 | 0.453 | 加 FFR 后 CAP 提升 0.116 |
| +FFR+AMC | Only Collision | 0.483 | 0.472 | 完整模型，SOTA |
| +FFR+AMC | Anomaly Onset | 0.484 | 0.479 | 使用人工标注起始帧 |

### 关键发现

- **FFR 贡献最大**：在 Only Collision 设置下，仅加 FFR 即可带来 CAP 上 0.116、Nexar 上 0.155 的 mAUC0.1 提升，证明未来帧正则化有效地传播了风险信号。
- **仅碰撞帧足矣**：完整模型在 Only Collision 下达到 0.483 (CAP) / 0.472 (Nexar)，与密集标注 (Anomaly Onset) 的 0.484 / 0.479 几乎持平，证明无需主观的异常起始标注。
- **风险曲线更平滑**：定性分析显示 RiskProp 在安全期保持低风险估计，仅在真正危险出现时急剧上升，有效抑制了传统方法常见的早期假阳性。

## 亮点与洞察

- **自监督链式传播机制非常巧妙**：通过 stop-gradient 让下一帧预测作为当前帧的目标，这种简单设计实现了从碰撞帧到早期帧的风险信号反向传播，无需额外的 teacher 模型或复杂架构。
- **"仅碰撞帧"匹配"密集标注"是核心亮点**：证明了在好的自监督约束下，极简标注可以达到密集标注的效果，对实际应用意义重大。
- **自适应单调约束的设计思想可迁移**：基于置信度和时间距离的自适应边距机制，可以推广到任何需要时序单调性约束的任务，如疾病进展预测、设备老化监控等。

## 局限与展望

- 碰撞帧标注仍然是必需的，完全无监督场景无法适用
- 非事故视频上 FFR 和 AMC 被禁用，意味着模型对安全场景的建模仅依赖 BCE 损失
- 仅验证了 3D CNN 编码器，未探索 Transformer 或多模态编码器
- 帧率固定重采样为 10 FPS，可能丢失快速变化场景的关键信息
- 可以考虑将风险传播扩展为双向（不仅从碰撞帧回传，也从安全期前传约束）

## 相关工作与启发

- **vs AdaLEA/CRASH**: 这些方法依赖指数衰减加权或手动异常起始标注来定义正样本窗口，RiskProp 完全移除了这些主观设计，仅用碰撞帧+自监督约束即超越它们
- **vs DSTA**: DSTA 在 CAP 上 mAUC 最高 (0.895)，但 RiskProp 在早期预警 (mAUC0.1, mTTA) 上大幅领先，体现了不同评价重点下的取舍
- 自监督时序传播思想可以启发视频异常检测、动作预测等相关领域

## 评分

- 新颖性: ⭐⭐⭐⭐ 碰撞锚定+自监督传播的范式新颖，但核心技术（stop-gradient 伪标签、单调约束）分别已有先例
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集、完整消融、三种标注策略对比、风险曲线可视化，非常充分
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，方法描述严谨，图表设计直观
- 价值: ⭐⭐⭐⭐ 减少标注依赖对实际部署有重要意义，风险曲线的可解释性是安全关键系统的加分项

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Interpretable Self-Supervised Learning via Representer Landmarks and Nyström Approximation](../../ICML2026/interpretability/interpretable_self-supervised_learning_via_representer_landmarks_and_nyström_app.md)
- [\[ICML 2026\] IdEst: Assessing Self-Supervised Learning Representations via Intrinsic Dimension](../../ICML2026/interpretability/idest_assessing_self-supervised_learning_representations_via_intrinsic_dimension.md)
- [\[ICCV 2025\] AIM: Amending Inherent Interpretability via Self-Supervised Masking](../../ICCV2025/interpretability/aim_amending_inherent_interpretability_via_self-supervised_masking.md)
- [\[CVPR 2025\] Probing the Mid-Level Vision Capabilities of Self-Supervised Learning](../../CVPR2025/interpretability/probing_the_mid-level_vision_capabilities_of_self-supervised_learning.md)
- [\[NeurIPS 2025\] Dataset Distillation for Pre-Trained Self-Supervised Vision Models](../../NeurIPS2025/interpretability/dataset_distillation_for_pre-trained_self-supervised_vision_models.md)

</div>

<!-- RELATED:END -->
