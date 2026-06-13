---
title: >-
  [论文解读] PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference
description: >-
  [ACL 2025][LLM对齐][safety alignment] 发布 PKU-SafeRLHF 大规模安全偏好数据集，包含 44.6k 精炼 prompt、265k 带安全元标签的 QA 对和 166.8k 偏好数据，首次引入 19 种危害类别和 3 级严重程度标注，并训练了严重程度敏感的审核模型（93% 准确率）和基于该数据的 SafeRLHF 对齐 pipeline。
tags:
  - "ACL 2025"
  - "LLM对齐"
  - "safety alignment"
  - "RLHF"
  - "preference data"
  - "harm categories"
  - "severity levels"
  - "moderation"
---

# PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference

## 基本信息

**会议**: ACL 2025  
**arXiv**: 2406.15513  
**数据集**: [PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)  
**机构**: Peking University / HKUST / Infinigence-AI  
**领域**: LLM 安全对齐 / RLHF  
**关键词**: safety alignment, RLHF, preference data, harm categories, severity levels, moderation  

## 一句话总结

发布 PKU-SafeRLHF 大规模安全偏好数据集，包含 44.6k 精炼 prompt、265k 带安全元标签的 QA 对和 166.8k 偏好数据，首次引入 19 种危害类别和 3 级严重程度标注，并训练了严重程度敏感的审核模型（93% 准确率）和基于该数据的 SafeRLHF 对齐 pipeline。

## 研究背景与动机

- **LLM 安全问题**：LLM 的训练数据来自互联网，包含大量噪声、错误和社会偏见，导致模型可能生成攻击性内容、泄露隐私、传播虚假信息，甚至展现欺骗性对齐行为
- **数据瓶颈**：安全对齐方法的有效性依赖于高质量的偏好数据和元标签分类，但大规模标注成本高昂
- **现有局限**：
    - 前作 BeaverTails 的 prompt 从互联网收集，质量和多样性受限，危害类别分布长尾
    - 缺乏**严重程度**分级（现有工作多为二元安全/不安全判断）
    - 有用性和无害性标注通常耦合，缺乏解耦标注
- **核心目标**：提供开源、高质量、多层次的安全偏好数据集，支持更精细的风险控制和更有效的安全对齐

## 方法详解

### 数据集构建

#### 数据规模

| 组成 | 规模 |
|------|------|
| 精炼 prompt | 44.6k |
| Q-A 对（含安全元标签） | 265k |
| 偏好数据 | 166.8k |
| 危害类别 | 19 种 |
| 严重程度级别 | 3 级 |

#### Prompt 生成

- 63.6% 由 Alpaca3-70B 生成，14% 由 WizardLM-30B-Uncensored 生成
- 为每个危害类别编写安全指南和 few-shot 示例
- 输入严重程度规则，要求生成从轻微到严重的三种不同 prompt
- 使用 Alpaca3-70B 添加上下文扩展 prompt，增强多样性

#### 回答生成

使用 Alpaca-(1,2,3) 模型生成回答：
1. 先用默认参数生成高质量回答
2. 提高温度生成 10 个额外回答
3. 基于文本相似度排序并过滤乱码，选取高质量低相似度回答
- 相比 BeaverTails，语义不清和乱码内容减少 **32%**

#### 模型选择策略

- 使用 Llama 系列基座模型（7B/8B）+ Alpaca 52K SFT
- **不使用 chat 模型或更大模型**的原因：
    - RLHF 需要 PTX loss，SFT 数据更透明
    - 7B/8B 参数量适合学术界在单机 8×A800 上训练

### 19 种危害类别

覆盖全面的安全风险维度，包括但不限于：侮辱行为、歧视行为、隐私侵犯、网络犯罪、经济犯罪、白领犯罪、精神操控等。

**类别间相关性分析**：
- 经济犯罪 ↔ 白领犯罪相关系数 0.55
- 侮辱行为 ↔ 歧视行为显著关联
- 大部分类别间低甚至负相关，说明分类体系有效

### 3 级严重程度定义

| 级别 | 描述 | 影响范围 |
|------|------|----------|
| **Minor（轻微）** | 短期、轻微负面影响，可自行恢复 | 个人 |
| **Moderate（中等）** | 通常违法，可能造成严重个人伤害或有限群体影响 | 个人→群体 |
| **Severe（严重）** | 针对群体，造成广泛严重伤害和长期影响 | 群体→社会 |

参考美国国会、MPAA 电影分级、FEMA 应急管理、PEGI 游戏分级和 Anthropic 负责任扩展政策。

### 标注方式：双偏好 + 单偏好

#### 安全元标签标注

- 28 名全职标注员，采用**人类+AI 联合标注**
- 标注内容：是否安全、所属危害类别（19种）、严重程度（3级）
- 相比 BeaverTails 的纯人类标注，一致性显著提高

#### 双偏好（Dual-Preference）标注

- **有用性偏好 $\mathcal{D}_R$**：对同一 prompt 的两个回答标注哪个更有用
- **无害性偏好 $\mathcal{D}_C$**：对同一 prompt 的两个回答标注哪个更无害
- 解耦标注使得可以独立训练 Reward Model 和 Cost Model

#### 单偏好（Single-Preference）标注

- 综合考虑有用性和无害性的整体偏好
- 直接从头衡量两者的权衡

### 应用一：严重程度敏感审核模型

利用全部严重程度元标签训练审核模型：

| 方法 | 安全准确率 | F1-Score | 误报率 |
|------|-----------|----------|--------|
| Llama-Guard | 0.78 | 0.71 | 0.055 |
| Llama-Guard 2 | 0.88 | 0.87 | 0.107 |
| Perspective API | 0.53 | 0.18 | 0.053 |
| OpenAI Moderation API | 0.53 | 0.10 | 0.002 |
| **Ours** | **0.93** | **0.93** | 0.077 |

- 二分类（安全/不安全）准确率 **93%**，显著超越所有基线
- 严重程度分类准确率 **85%**
- 19 种危害类别精确匹配准确率 **71.3%**

### 应用二：Safe RLHF Pipeline

#### Reward Model (RM)

使用 Bradley-Terry 模型训练有用性奖励模型：

$$\mathcal{L}_R(\phi; \mathcal{D}_R) = -\mathbb{E}[\log \sigma(R_\phi(y_w, x) - R_\phi(y_l, x))]$$

#### Cost Model (CM)

除了配对比较损失，还引入分类项以利用安全标签信息：

$$\mathcal{L}_C(\psi; \mathcal{D}_C) = -\mathbb{E}[\log \sigma(s_w \cdot C_\psi(y_w, x)) + \log \sigma(s_l \cdot C_\psi(y_l, x))]$$

其中 $s(y) = +1$（有害）或 $-1$（无害）。

## 实验

### RLHF 对齐实验（表2）

| 数据集/设置 | Alpaca1 有用性 | Alpaca1 无害性 | Alpaca2 无害性 | Alpaca3 无害性 |
|------------|--------------|--------------|--------------|--------------|
| BeaverTails (dual) | 76.8% | 83.7% | 63.8% | 77.1% |
| Ours (single) | 81.4% | 86.1% | 88.6% | 86.8% |
| **Ours (dual)** | **87.3%** | **86.5%** | **94.0%** | **92.5%** |

**关键发现**：
1. 双偏好（解耦有用性和无害性）显著优于单偏好直接对齐
2. PKU-SafeRLHF 数据质量优于 BeaverTails

### 直接对比实验（表3）

PKU-SafeRLHF 对齐模型 vs Alpaca 原始模型的胜率：
- 有用性胜率：80.86% ~ 90.25%
- 无害性胜率：86.50% ~ 92.33%

### RM/CM 评估验证

- RM 和 CM 的评估与人类评估的一致性高（图7a）
- CM 的安全阈值（分数=0）与人类标注的安全边界（3~4分）高度吻合
- 验证了 CM 可作为可靠的逐点评估指标（尽管是通过配对排序损失训练的）

## 亮点与洞察

1. **多层次安全标注**：首次在安全偏好数据中引入 19 种危害类别 + 3 级严重程度，超越二元安全判断
2. **解耦标注设计**：有用性和无害性的双偏好标注使研究者能独立研究两个维度及其权衡
3. **高质量数据**：人类+AI联合标注显著提高一致性；精心设计的数据生成流程减少 32% 的噪声
4. **实用的审核模型**：严重程度敏感的审核模型可以精细化控制不同风险等级，适用于实际部署
5. **CM 的标定性质**：Cost Model 虽用配对损失训练，但其分数与人类严重程度评估高度一致，可直接用于逐点评估

## 局限性

1. **数据规模**：相比商业组织的大规模偏好数据集，规模较小（166.8k）
2. **类别重叠**：19 种危害类别间存在不可避免的重叠（如经济犯罪↔白领犯罪）
3. **领域适应性有限**：虽然覆盖广泛，但法律、医疗、金融等高风险领域需要领域特定标注
4. **文化和语言局限**：聚焦英语，非英语和不同文化背景下的适用性有限
5. **标注者偏差**：尽管采用联合标注，28 名标注者的个体差异仍可能影响一致性

## 相关工作

- **安全数据集**：BeaverTails (Ji et al., 2024)、PKU-Beaver (Dai et al., 2023)
- **对齐方法**：RLHF (Ouyang et al., 2022)、SafeRLHF (Dai et al., 2024)、DPO (Rafailov et al., 2024)
- **安全审核**：Llama-Guard (Inan et al., 2023)、Perspective API、OpenAI Moderation API
- **安全评估**：Red Teaming (Zhu et al., 2023)、Anthropic Responsible Scaling Policy

## 评分

⭐⭐⭐⭐ (4/5)

- **数据价值**：高质量、多层次的安全偏好数据集，填补了精细化安全标注的空白（+1）
- **实用性**：审核模型和 RLHF pipeline 可直接用于实际应用（+0.5）
- **标注设计**：双偏好+严重程度的创新标注方案有理论和实践意义（+0.5）
- **开源贡献**：数据集和模型完全开源，有利于社区研究（+0.5）
- **扣分**：数据规模相对有限、危害类别间重叠难以完全消除、仅限英语（-1）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MTSA: Multi-Turn Safety Alignment for LLMs through Multi-Round Red-Teaming](mtsa_multi-turn_safety_alignment_for_llms_through_multi-round_red-teaming.md)
- [\[CVPR 2025\] Do We Really Need Curated Malicious Data for Safety Alignment in Multi-Modal LLMs?](../../CVPR2025/llm_alignment/do_we_really_need_curated_malicious_data_for_safety_alignment_in_multi-modal_lar.md)
- [\[ACL 2025\] AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs](automixalign_adaptive_data_mixing.md)
- [\[ACL 2025\] AgentAlign: Navigating Safety Alignment in the Shift from Informative to Agentic LLMs](agentalign_navigating_safety_alignment_in_the_shift_from_informative_to_agentic_.md)
- [\[ACL 2025\] LSSF: Safety Alignment via Low-Rank Safety Subspace Fusion](lssf_safety_subspace.md)

</div>

<!-- RELATED:END -->
