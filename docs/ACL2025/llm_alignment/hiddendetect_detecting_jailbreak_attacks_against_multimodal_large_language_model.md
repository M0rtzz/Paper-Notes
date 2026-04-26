---
title: >-
  [论文解读] HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States
description: >-
  [ACL 2025][LLM对齐][越狱攻击检测] 提出 HiddenDetect，一种免训练（tuning-free）的基于内部激活状态的安全检测框架：通过监控 LVLM 推理时隐藏状态中的拒绝语义信号来检测越狱攻击，在多个模型和多模态基准上 AUROC 大幅超越现有方法。
tags:
  - ACL 2025
  - LLM对齐
  - 越狱攻击检测
  - 多模态安全
  - 隐藏状态分析
  - LVLM
  - 免训练防御
  - 拒绝语义
---

# HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States

**会议**: ACL 2025  
**arXiv**: [2502.14744](https://arxiv.org/abs/2502.14744)  
**代码**: [GitHub](https://github.com/leigest519/HiddenDetect)  
**领域**: llm_alignment  
**关键词**: 越狱攻击检测, 多模态安全, 隐藏状态分析, LVLM, 免训练防御, 拒绝语义

## 一句话总结

提出 HiddenDetect，一种免训练（tuning-free）的基于内部激活状态的安全检测框架：通过监控 LVLM 推理时隐藏状态中的拒绝语义信号来检测越狱攻击，在多个模型和多模态基准上 AUROC 大幅超越现有方法。

## 研究背景与动机

**领域现状**：大型视觉语言模型（LVLMs）如 GPT-4V、LLaVA 等通过整合视觉与语言模态，在视觉推理、问答等任务上表现出色。但多模态能力也引入了新的安全风险——相比纯文本 LLM，LVLMs 更容易受到对抗性操纵。

**现有痛点**：已有安全防御主要依赖行为级手段：(a) 在安全数据集上微调——计算开销大且不够灵活；(b) 手工设计安全提示词——常导致过度防御，降低模型实用性；(c) 推理时推理模块（如多模态 CoT）——无法泛化到未见攻击策略。这些方法都是"事后补救"，忽略了模型内部可能已编码的安全信号。

**核心矛盾**：LVLMs 在处理不安全输入时，其内部是否已经存在可利用的安全信号？如果存在，为什么多模态输入（尤其是对抗性图像）仍能成功绕过安全机制？

**本文目标**：(1) 揭示 LVLMs 隐藏状态中是否编码了安全相关信号；(2) 分析跨模态和跨层的安全信号变化规律；(3) 利用这些内在信号构建高效的越狱检测框架。

**切入角度**：从可解释性视角出发，构建"拒绝向量"（Refusal Vector）来量化隐藏状态与拒绝语义的对齐程度，并发现安全信号在中间层最为显著、在视觉输入下存在延迟激活的现象。

**核心 idea**：LVLMs 内部已编码安全信号，通过监控安全敏感层的拒绝语义强度即可无训练地检测越狱攻击。

## 方法详解

### 整体框架

HiddenDetect 的流程分为两个阶段：

1. **离线阶段**：构建拒绝向量 + 识别安全敏感层（仅需少量样本）
2. **在线检测**：对新输入计算各层拒绝语义强度，聚合安全得分，判断是否安全

### 关键设计

#### 1. 构建多模态拒绝向量（Refusal Vector）

**功能**：在词汇空间构建一个稀疏二值向量 $\mathbf{r}$，标记所有与拒绝行为相关的 token。

**为什么**：不同于先前 LLM 工作仅从纯文本中挖掘拒绝信号，LVLMs 的拒绝行为受视觉语义调节，需要在多模态上下文中捕获拒绝 token。

**怎么做**：
- 收集模型对有害图文对的响应，提取高频拒绝 token（如 "sorry"、"unable"、"cannot"）构成初始拒绝 token 集（RTS）
- 将各层最后 token 位置的隐藏状态通过 unembedding 层投影到词汇空间，收集 top-5 logits 对应的 token，补充到 RTS 中
- 迭代直到无新 token 加入，最终 RTS 构成拒绝向量 $\mathbf{r}$（对应位置为 1，其余为 0）

#### 2. 逐层安全意识评估（Refusal Strength Vector）

**功能**：量化每一层隐藏状态与拒绝语义的对齐程度。

**为什么**：安全信号并非均匀分布在所有层——需要找到"安全敏感层"以实现精准检测。

**怎么做**：对 few-shot 的安全/不安全样本对，将各层的最后 token 隐藏状态投影到词汇空间，计算与拒绝向量的余弦相似度：

$$F_l = \cos(\mathbf{h}_l, \mathbf{r}), \quad l \in \{0, 1, \ldots, L-1\}$$

然后计算"拒绝差异向量"（Refusal Discrepancy Vector）：

$$F' = F_{\text{unsafe}} - F_{\text{safe}}$$

$F'_l > 0$ 的层即为安全敏感层。实验发现安全信号在中间层最强，后期因平衡响应相关性而衰减。

#### 3. 安全敏感层范围识别

**功能**：自动确定用于检测的层范围 $(s, e)$。

**怎么做**：以最后一层的差异分数 $F'_{L-1}$ 作为保守基线，选取所有超过此基线的层：

$$s = \min\{l : F'_l > F'_{L-1}\}, \quad e = \max\{l : F'_l > F'_{L-1}\}$$

#### 4. 安全评分与检测

**功能**：对输入 prompt 计算安全得分，超过阈值即判定为不安全。

**怎么做**：对安全敏感层范围内的拒绝强度用梯形法则计算 AUC 作为安全得分：

$$s(F) = \text{AUC}_{\text{trapezoid}}(\{F_l : l \in \mathcal{L}_\mathcal{M}\})$$

### 关键发现：多模态越狱的延迟安全激活

论文进一步分析了五类有害查询的拒绝差异向量：
- 纯文本有害查询的安全激活最早、最强
- 即使文本本身有害，配对图像也会削弱早期和中间层的安全激活
- 仅通过图像传递有害意图（文本无害）时，安全激活严重延迟且大幅减弱
- 双模态安全对齐（bimodal alignment）可显著增强跨模态查询的拒绝强度，对纯文本查询几乎无影响

## 实验关键数据

### 主实验（Table 1: AUROC 检测性能）

| 方法 | 训练免 | XSTest | FigTxt | MM-SafetyBench | FigImg | JailBreakV-28K |
|------|--------|--------|--------|----------------|--------|----------------|
| **LLaVA** | | | | | | |
| Perplexity | ✗ | 0.610 | 0.758 | 0.825 | 0.683 | 0.781 |
| GradSafe | ✓ | 0.714 | 0.831 | 0.889 | 0.760 | 0.845 |
| JailGuard | ✗ | 0.662 | 0.784 | 0.859 | 0.715 | 0.801 |
| **HiddenDetect** | **✓** | **0.868** | **0.976** | **0.997** | **0.846** | **0.932** |
| **Qwen-VL** | | | | | | |
| GradSafe | ✓ | 0.678 | 0.809 | 0.872 | 0.744 | 0.839 |
| **HiddenDetect** | **✓** | **0.834** | **0.962** | **0.991** | **0.823** | **0.907** |
| **CogVLM** | | | | | | |
| GradSafe | ✓ | 0.617 | 0.762 | 0.812 | 0.692 | 0.789 |
| **HiddenDetect** | **✓** | **0.762** | **0.866** | **0.910** | **0.764** | **0.883** |

- HiddenDetect 在所有 3 个模型、5 个数据集上均取得最高 AUROC
- 相比最强基线 GradSafe，平均 AUROC 提升约 11.1 个百分点（0.777 → 0.888）
- 在 MM-SafetyBench 上接近完美检测（LLaVA: 0.997, Qwen-VL: 0.991）

### 消融实验（Table 2: 安全敏感层的作用）

| 设置 | FigTxt | FigImg | MM-SafetyBench |
|------|--------|--------|----------------|
| 不含安全敏感层 | 0.630 | 0.502 | 0.750 |
| 使用全部层 | 0.861 | 0.640 | 0.960 |
| **仅用安全敏感层** | **0.925** | **0.830** | **0.977** |

- 移除安全敏感层后检测性能骤降，尤其 FigImg 从 0.830 降至 0.502
- 使用全部层引入噪声，反而不如精确选择安全敏感层

### 安全层权重缩放实验（Table 3）

缩放 LLaVA 第 16-29 层的权重，缩放因子越大，被拒绝的样本数越多（从基线 33 增至 49），进一步验证这些层对安全决策的关键作用。

### 关键发现

1. 拒绝语义在中间层（约 16-29 层）最为显著，后期因解码需求而衰减
2. 视觉输入导致安全激活延迟——这解释了为什么多模态攻击比纯文本攻击更容易成功
3. 该方法对纯文本越狱攻击同样有效，具有跨模态泛化能力

## 亮点与洞察

- **免训练设计**：无需微调或外部分类器，直接利用模型内在的安全信号，计算开销极低
- **首次系统分析 LVLM 安全信号的跨层跨模态演化**：发现了"视觉输入延迟安全激活"这一重要现象，为理解多模态越狱提供了新视角
- **拒绝向量的多模态构建**：区别于前人仅从文本构建拒绝方向，本工作在视觉条件下挖掘拒绝 token，更适配 LVLM 场景
- **few-shot 即可使用**：仅需少量安全/不安全样本即可定位安全敏感层，无需大规模标注数据
- **可视化分析直观有力**：将隐藏状态投影到"拒绝语义-正交语义"平面，清晰展示了安全信号从混合到分离的演化过程

## 局限与展望

1. **假设不安全 prompt 一定导致不同的激活模式**：精心设计的对抗输入可能在潜空间中规避检测，尤其在安全信号的决策边界附近
2. **仅做"检测"不做"干预"**：HiddenDetect 标记不安全 prompt 但不修正模型输出，可结合激活引导（activation steering）实现响应级纠正
3. **拒绝 token 集的构建依赖人工启发**：初始种子 token 来自经验选择，可能遗漏某些模型特有的拒绝表达
4. **安全敏感层范围 $(s, e)$ 为静态**：对不同攻击类型可能需要动态调整层范围
5. **仅在开源模型上验证**：未在闭源 API 模型上测试（受限于无法获取中间层激活）
6. **阈值设定需人工调优**：安全得分的判定阈值是可配置的，但论文未充分讨论如何在不同部署场景下自适应设定

## 相关工作与启发

### 与 GradSafe (Xie et al., 2024) 的对比
GradSafe 基于梯度信息检测不安全输入，也无需训练，但依赖对反向传播的访问。HiddenDetect 仅需前向传播的中间激活，计算更轻量，且在所有基准上大幅超越 GradSafe。

### 与 Refusal Direction (Arditi et al., 2024) 的关系
Arditi 等人发现 LLM 的拒绝行为由单一方向介导，但仅在纯文本 LLM 上验证。HiddenDetect 将这一思路扩展到多模态场景，发现 LVLM 中拒绝信号的层级分布更复杂、受模态影响更大。

### 与 CIDER / MirrorCheck 等变异/去噪方法的对比
这些方法通过变异输入或镜像检查来判断安全性，需要多次推理。HiddenDetect 单次前向传播即可完成检测，效率优势明显。

### 启发
- 拒绝语义的层级分布分析可推广到其他安全相关任务（如偏见检测、幻觉检测）
- "视觉输入延迟安全激活"这一发现可指导 LVLM 安全对齐训练——应特别强化中间层对多模态有害输入的响应

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统分析 LVLM 内部安全信号并用于检测，"延迟激活"发现有启发性
- 实验充分度: ⭐⭐⭐⭐ — 3 个模型 × 5 个数据集 × 7 个基线，消融和可视化充分；但缺少更大模型和闭源模型测试
- 写作质量: ⭐⭐⭐⭐ — 从发现到方法的论述逻辑清晰，可视化丰富；公式符号偶有不一致
- 价值: ⭐⭐⭐⭐⭐ — 免训练、轻量、泛化性强，对 LVLM 安全部署有直接实用价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] AGD: Adversarial Game Defense Against Jailbreak Attacks in Large Language Models](agd_adversarial_game_defense_against_jailbreak_attacks_in_large_language_models.md)
- [\[ACL 2025\] JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs](jailbreakradar_comprehensive_assessment_jailbreak_attacks.md)
- [\[ACL 2025\] Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking](red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)
- [\[ACL 2025\] Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs](beyond_surface-level_patterns_an_essence-driven_defense_framework_against_jailbr.md)
- [\[ACL 2025\] Beyond the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models](beyond_the_tip_of_efficiency_uncovering_the_submerged_threats_of_jailbreak_attac.md)

<!-- RELATED:END -->
