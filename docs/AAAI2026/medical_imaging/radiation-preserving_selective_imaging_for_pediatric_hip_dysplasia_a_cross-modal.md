---
title: >-
  [论文解读] Radiation-Preserving Selective Imaging for Pediatric Hip Dysplasia: A Cross-Modal Approach
description: >-
  [AAAI 2026][医学图像][发育性髋关节发育不良] 提出一种"超声优先、保辐射"的跨模态选择性成像策略，通过自监督预训练的冻结编码器、测量忠实的轻量头网络和共形预测校准的单侧下界，实现了在发育性髋关节发育不良（DDH）诊断中有据可依地决定何时仅用超声即可、何时需要额外的 X 光检查。
tags:
  - "AAAI 2026"
  - "医学图像"
  - "发育性髋关节发育不良"
  - "超声-X光跨模态"
  - "共形预测"
  - "选择性成像"
  - "自监督学习"
---

# Radiation-Preserving Selective Imaging for Pediatric Hip Dysplasia: A Cross-Modal Approach

**会议**: AAAI 2026 Best Paper Oral  
**arXiv**: [2511.18457](https://arxiv.org/abs/2511.18457)  
**代码**: 无  
**领域**: 医学影像 / 跨模态学习 / 选择性成像  
**关键词**: 发育性髋关节发育不良, 超声-X光跨模态, 共形预测, 选择性成像, 自监督学习

## 一句话总结

提出一种"超声优先、保辐射"的跨模态选择性成像策略，通过自监督预训练的冻结编码器、测量忠实的轻量头网络和共形预测校准的单侧下界，实现了在发育性髋关节发育不良（DDH）诊断中有据可依地决定何时仅用超声即可、何时需要额外的 X 光检查。

## 研究背景与动机

**发育性髋关节发育不良（DDH）**是婴幼儿时期常见的骨科疾病，涵盖从髋臼覆盖不足到髋关节脱位的一系列病变。临床上存在两种互补的影像检查方式：

**超声（US）**：用于早期筛查和管理，报告 Graf α/β 角和股骨头覆盖率。优点是无辐射、可及性强。

**X 光（XR）**：在骨化进展后更适合评估髋臼发育和手术规划，测量髋臼指数（AI）、中心边缘角（CE）和 IHDI 分级。缺点是有电离辐射。

**核心矛盾**：在儿科中，最小化电离辐射暴露是重要原则，但单纯依赖超声可能会遗漏需要 X 光才能发现的异常。临床上，**明显正常的超声**（高 α 角、充分覆盖）通常意味着 X 光的额外信息价值低；异常或边界的超声则提高了 X 光的价值。

**现有工作的不足**：
- 超声自动化和 X 光自动化已分别取得进展（自动检测标准平面、预测 Graf 测量值、AI/CE 角等），但**跨模态 US-XR 学习极为稀缺**。
- 缺乏一种实用的、基于测量值的策略来权衡辐射风险与诊断收益，并提供明确的统计保证。
- 现有方法多为黑盒分类器，缺乏与临床决策阈值的直接对接。

本文的动机是填补这一空白：**构建一个可校准、可解释、可调节的"超声优先"策略，通过共形预测提供有限样本分布无关的覆盖保证，让临床团队能够在辐射暴露和遗漏风险之间做出透明的权衡。**

## 方法详解

### 整体框架

本文提出了一个四阶段流水线：

1. **自监督预训练**：在大规模无标注注册数据（37,186 张超声 + 19,546 张 X 光）上分别用 SimSiam 预训练模态特定的 ResNet-18 编码器，然后冻结。
2. **测量忠实头网络**：在冻结编码器上训练轻量 MLP 头，直接预测临床使用的命名测量值（α/β 角、覆盖率、AI、CE、IHDI 等）。
3. **共形校准**：在校准集上拟合仿射偏差校正并计算单侧共形残差分位数，为超声预测值提供有限样本的边际覆盖保证。
4. **选择性成像策略**：基于校准后的超声下界与临床阈值的比较，决定"仅超声"还是"转 X 光"，并通过决策曲线分析权衡辐射成本与遗漏惩罚。

### 关键设计

1. **SimSiam 自监督预训练**：对每种模态独立训练编码器 $f_\phi$。设投影 MLP 为 $h_\phi$，预测 MLP 为 $q_\phi$。对图像 $x$ 生成两个增强视图 $v_1, v_2$，损失函数为停止梯度负余弦相似度：$\mathcal{L}_{SSL} = -\frac{1}{2}[\frac{\langle p_1, \text{sg}(z_2)\rangle}{\|p_1\| \|\text{sg}(z_2)\|} + \frac{\langle p_2, \text{sg}(z_1)\rangle}{\|p_2\| \|\text{sg}(z_1)\|}]$。训练 10 epoch 后丢弃投影和预测头，冻结编码器。设计动机：利用大量无标注影像学习通用视觉表示，弥补有标注数据稀缺的不足。

2. **测量忠实头网络**：从冻结编码器提取 512 维全局平均池化特征 $u = f_\phi(x)$，在其上训练小型 MLP：

    - **超声头**：单隐层（128 单元），3 输出预测 $\hat{\alpha}, \hat{\beta}, \widehat{\text{cov}}$，损失 $\mathcal{L}_{US} = \lambda_\alpha |\hat{\alpha} - y^\alpha| + \lambda_\beta |\hat{\beta} - y^\beta| + \lambda_{\text{cov}} |\widehat{\text{cov}} - y^{\text{cov}}|$
    - **X 光头**：预测 AI 和 CE 角（MAE 损失）加可选的 IHDI 分类（交叉熵）
   核心设计理念是**测量忠实（measurement-faithful）**——输出的是临床医生日常使用的命名测量值，而非黑盒分类，使决策过程可追溯。

3. **共形校准的单侧下界**：分为两步：

    - (i) **仿射偏差校正**：在校准集上拟合 $\tilde{y}^t = a_t \hat{y}^t + b_t$，减少系统性偏差而不重训头网络。
    - (ii) **单侧残差分位数**：定义残差 $r_i^t = y_i^t - \tilde{y}_i^t$，对误覆盖水平 $\delta_t$ 计算共形半径 $q_t^+(\delta_t)$，确保在可交换性假设下以 $\geq 1 - \delta_t$ 的概率满足 $y^t \geq \text{LB}_t(x; \delta_t)$。校准下界为 $\text{LB}_t(x; \delta_t) = \tilde{y}^t(x) - q_t^+(\delta_t)$。

4. **三种选择性成像规则**：基于校准下界与临床阈值 $T_\alpha = 60°$, $T_{\text{cov}} = 50\%$ 的比较：

    - **Alpha-only**：$d_\alpha(x) = \mathbb{I}[\text{LB}_\alpha(x) \geq 60°]$
    - **Alpha OR Coverage**：任一下界超过阈值即可仅用超声
    - **Alpha AND Coverage**：两者同时超过阈值才可仅用超声
   通过扫描 $(\delta_\alpha, \delta_{\text{cov}})$ 网格生成一系列策略，保守设置（小 $\delta$）提供高覆盖但低超声通过率，宽松设置（大 $\delta$）增加超声通过率但接受更高遗漏风险。

### 损失函数 / 训练策略

- 自监督阶段：SimSiam 负余弦相似度损失，10 epoch/模态
- 有监督头网络：超声用 MAE 损失（等权 $\lambda_\alpha = \lambda_\beta = \lambda_{\text{cov}} = 1$），X 光用 MAE + 交叉熵
- 仅在 post-train 集（30 subjects, 136 images）上训练，编码器始终冻结
- 校准集（7 subjects, 28 images）仅用于偏差校正和共形校准，不参与训练
- 数据划分严格按 subject 级别防止泄漏

## 实验关键数据

### 主实验

数据集：从大型注册库中配对的 75 个受试者、321 张图像。评估集 38 subjects（157 images），严格匹配后得到 N=77 个髋关节配对。

| 模态 | 测量指标 | MAE | 说明 |
|------|---------|:---:|------|
| 超声 | α 角 | 9.69° | 冻结编码器 + 轻量头 |
| 超声 | β 角 | 11.25° | 同上 |
| 超声 | 股骨头覆盖率 | 13.97 pp | 百分点 |
| X 光 | 髋臼指数 (AI) | 7.60° | 冻结编码器 + 轻量头 |
| X 光 | 中心边缘角 (CE) | 8.93° | 同上 |

**选择性成像策略结果**（N=77 严格配对）：

| 规则 | 误覆盖 $\delta_\alpha / \delta_{\text{cov}}$ | 超声通过率 | X 光使用率 |
|------|:---:|:---:|:---:|
| AND | 0.10 / 0.10 | 0.00 | 1.00 |
| AND | 0.20 / 0.20 | 0.00 | 1.00 |
| OR | 0.35 / 0.35 | 0.43 | 0.57 |
| OR | 0.40 / 0.40 | 0.55 | 0.45 |

### 消融实验

| 配置 | 关键结果 | 说明 |
|------|---------|------|
| 保守设置 AND 0.10/0.10 | 覆盖率 ~0.90 (α), ~0.94 (cov) | 安全但几乎全部转 X 光 |
| 宽松设置 OR 0.40/0.40 | 超声通过率 55% | 大幅减少 X 光但遗漏风险上升 |
| 校准半径 (δ=0.10) | $q_\alpha^+ = 10.75°$, $q_{\text{cov}}^+ = 28.74$ pp | 保守下界 |
| 决策曲线分析 | 辐射成本 λ 高时 OR 策略最优 | 显式权衡辐射 vs 安全 |

### 关键发现

1. **冻结编码器 + 轻量头即可竞争**：仅用 30 个受试者训练的小 MLP 头在冻结的自监督编码器上即可达到与使用更大网络的单模态方法可比的测量精度。
2. **共形覆盖遵循理论预期**：经验覆盖率随误覆盖水平 $\delta$ 单调变化，验证了有限样本保证的有效性。
3. **三种规则族提供灵活性**：AND 规则最保守（几乎全部转 X 光），OR 规则最宽松（可节省约一半 X 光），Alpha-only 介于两者之间。
4. **边界病例自动转 X 光**：许多边界髋关节的 α 角在 60° 阈值附近几度范围内，系统不会勉强用超声决策这些病例，而是转向 X 光，这与经验丰富的临床医生在不确定时的行为一致。

## 亮点与洞察

- **测量忠实而非黑盒**：整个推理链可见——线和点标注 → 导出测量值 → 校准下界 → 规则评估 → 策略结果。没有隐藏的 logits 或不透明的多类输出来控制安全关键决策，这对临床部署至关重要。
- **标签高效流水线**：自监督预训练利用大量无标注数据学习表示，冻结编码器后仅需极少标注（30 subjects）训练头网络，完美适配医学影像中标注昂贵的现实。
- **共形预测提供分布无关保证**：不依赖于数据分布假设，在可交换性条件下即可提供有限样本覆盖控制，这比传统的置信区间更稳健。
- **决策曲线显式暴露权衡**：通过辐射成本 $\lambda$ 和遗漏惩罚 $\mu$ 两个参数，将建模与政策分离，让辐射管理权掌握在临床团队手中。
- **安全失效（fail-safe）设计**：对于不确定的病例自动转向 X 光而非冒险，单侧共形下界保证了覆盖控制。

## 局限与展望

1. **数据规模小**：标注数据仅 75 个受试者，校准集仅 7 个受试者（26 张超声），共形半径可能较宽。需要更大规模的多中心验证。
2. **标注者为实习生**：测量标注由 trainee 完成而非资深专家，可能引入标注噪声。
3. **未包含年龄/骨化感知阈值**：虽然预留了骨化核标志接口，但未实际实现年龄相关的差异化阈值。
4. **缺少前瞻性临床研究**：需要在实际临床工作流中验证每 100 名婴儿节省的辐射量、回访率变化等。
5. **跨模态代理模型未完成**：论文提到可以从超声测量预测 X 光的 AI 或 IHDI 风险作为代理指标，但未实现这一扩展。
6. **覆盖率 MAE 较高**：股骨头覆盖率的 MAE 约 14 个百分点，仍有提升空间。

## 相关工作与启发

本文是一个将**统计决策理论**（共形预测、决策曲线分析）应用于**临床影像工作流优化**的优秀范例。它不追求单一模态上的极致精度，而是将跨模态信息整合为一个**可操作的选择性决策框架**。这种方法论可以推广到其他"昂贵检查是否值得做"的临床场景，如：先做低成本筛查（血液检查、简单影像），再决策是否需要昂贵或有风险的进一步检查（造影、活检等）。共形预测在医学AI中的应用正在兴起，本文展示了如何将其与临床决策融合的实际路径。

## 评分

- 新颖性：⭐⭐⭐⭐ — 首个基于共形预测的跨模态选择性成像策略
- 实验充分性：⭐⭐⭐ — 数据规模小，缺乏外部验证和前瞻性研究
- 实用性：⭐⭐⭐⭐ — 临床工作流高度可操作，决策透明可审计
- 写作质量：⭐⭐⭐⭐⭐ — 方法论阐述极为清晰，临床动机与技术方案紧密结合

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Learning with Preserving for Continual Multitask Learning](learning_with_preserving_for_continual_multitask_learning.md)
- [\[AAAI 2026\] FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)
- [\[AAAI 2026\] Bidirectional Channel-selective Semantic Interaction for Semi-Supervised Medical Segmentation](bidirectional_channel-selective_semantic_interaction_for_semi-supervised_medical.md)
- [\[ICML 2025\] Enhancing Statistical Validity and Power in Hybrid Controlled Trials: A Randomization Inference Approach with Conformal Selective Borrowing](../../ICML2025/medical_imaging/enhancing_statistical_validity_and_power_in_hybrid_controlled_trials_a_randomiza.md)
- [\[CVPR 2026\] CRFT: Consistent-Recurrent Feature Flow Transformer for Cross-Modal Image Registration](../../CVPR2026/medical_imaging/crft_consistent-recurrent_feature_flow_transformer_for_cross-modal_image_registr.md)

</div>

<!-- RELATED:END -->
