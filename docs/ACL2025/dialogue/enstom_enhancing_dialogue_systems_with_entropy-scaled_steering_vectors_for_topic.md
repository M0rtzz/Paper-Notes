---
title: >-
  [论文解读] EnSToM: Enhancing Dialogue Systems with Entropy-Scaled Steering Vectors for Topic Maintenance
description: >-
  [ACL 2025][对话系统] 提出 EnSToM，一种基于熵缩放转向向量的轻量级方法，通过利用 LLM 内部层级熵分布差异来动态调整转向强度，在不修改模型参数的情况下提升任务导向对话系统的主题维持能力。
tags:
  - ACL 2025
  - 对话系统
  - 主题一致性
  - 转向向量
  - 熵缩放
  - 激活工程
---

# EnSToM: Enhancing Dialogue Systems with Entropy-Scaled Steering Vectors for Topic Maintenance

**会议**: ACL 2025  
**arXiv**: [2505.16526](https://arxiv.org/abs/2505.16526)  
**代码**: https://github.com/linkyouhj/enstom (有)  
**领域**: 文本生成  
**关键词**: 对话系统, 主题一致性, 转向向量, 熵缩放, 激活工程

## 一句话总结

提出 EnSToM，一种基于熵缩放转向向量的轻量级方法，通过利用 LLM 内部层级熵分布差异来动态调整转向强度，在不修改模型参数的情况下提升任务导向对话系统的主题维持能力。

## 研究背景与动机

**领域现状**：sLLM（小型大语言模型）因轻量高效适合资源受限环境部署。企业级任务导向对话系统（如银行客服机器人）需要模型严格遵守预定主题，拒绝离题或恶意输入。

**现有痛点**：(1) sLLM 容量有限，在长时间交互中难以维持场景一致性；(2) **微调**方法需要大量数据和计算资源，难以覆盖所有场景；(3) **提示工程**在复杂场景中效果有限；(4) 直接应用**转向向量（Steering Vector）**虽能提高离题拒绝率，但会严重损害正常主题响应质量（on-topic 准确率从 0.94 降至 0.70）。

**核心矛盾**：转向向量能有效提升 distractor 拒绝能力，但无差别地对所有输入施加转向会导致 on-topic 响应也被错误拒绝——如何让转向"看人下菜碟"？

**本文目标**：设计一种自适应的转向强度调节机制，对 distractor 强力转向、对 on-topic 轻柔或不转向。

**切入角度**：发现 LLM 内部不同层的熵分布在 on-topic 和 distractor 输入之间存在显著差异，可作为区分信号来动态调节转向系数。

**核心 idea**：利用 LLM 层级生成熵区分离题/正题输入，通过 sigmoid 函数动态缩放转向向量强度，实现精准的主题维持。

## 方法详解

### 整体框架

EnSToM 由三个组件构成：(1) 从对比数据提取转向向量；(2) 基于熵的系数缩放动态调整转向强度；(3) 使用缩放后的转向向量生成响应。整个过程无需训练，纯推理时干预。

### 关键设计

1. **转向向量提取**：构建 Steering QA Dataset $S = \{q_1, q_2, \dots\}$，每个 $q_i$ 包含期望行为（拒绝并引导回主题）和非期望行为（继续回答离题问题）的对比提示。在指定层 $l$ 进行前向传播，计算期望与非期望行为的隐藏表示差值：
   $$v_s^i = h_p^{(l)} - h_n^{(l)}$$
   经归一化和平均得到最终转向向量 $v = \frac{1}{k}\sum_{i=1}^{k} \text{norm}(v_s^i)$。

2. **层级熵分析**：在 LLM 的第 $l$ 层计算生成前 2 个 token 的熵：
   $$E^{(l)} = \mathbb{E}\left[-\sum_{i=1}^{V} p_i^{(l)} \log(p_i^{(l)} + \epsilon)\right]$$
   关键发现：在 **Layer 16**（语义关键层），distractor 输入熵**低于** on-topic（因为离题内容引起高度聚焦的注意力）；在 **Layer 19**（深层），关系反转。

3. **熵缩放系数**：使用 sigmoid 函数将熵映射为转向系数：
   $$C_H^{(L)} = \frac{C_{\max}}{1 + e^{-\alpha \delta (H^{(L)} - t)}}$$
   其中 $C_{\max} = 1.5$ 为最大系数，$\alpha = 5$ 控制 sigmoid 陡度，$t = 7.5$ 为阈值，$\delta$ 根据熵分布方向取 $\pm 1$。distractor 输入获得高系数（强转向），on-topic 获得低系数（弱/无转向）。

4. **响应生成**：在推理时，先生成 2 个 token 计算熵、得到系数，再将缩放后的转向向量加到指定层的激活上：
   $$h'^{(l)} = h^{(l)} + C_H^{(L)} \cdot v$$

### 训练策略

- **完全无需训练**：仅需约 100 个对比样本提取转向向量
- 拒绝和响应选项由 GPT-4o 生成，随机分配位置避免位置偏差
- 评估使用 GPT-4o 分类模型响应为拒绝/回应

## 实验关键数据

### 主实验（LLaMA-2-7B-Chat，CantTalkAboutThis 银行领域）

| 方法 | 熵层 L | 转向层 | Distractor ↑ | On-topic ↑ | Overall ↑ |
|------|--------|--------|-------------|-----------|----------|
| Prompt Only | - | - | 0.282 | 0.938 | 0.610 |
| Vanilla Steering | - | - | 0.800 | 0.700 | 0.750 |
| EnSToM | 16 | 15 | 0.810 (+0.53) | 0.747 (-0.19) | 0.779 |
| **EnSToM** | **16** | **16** | **0.709 (+0.43)** | **0.895 (-0.04)** | **0.802** |
| EnSToM | 19 | 16 | 0.749 (+0.47) | 0.818 (-0.12) | 0.784 |

最佳配置（L=16, Steer@16）：overall 0.802，比 Prompt Only 高 19.2 个百分点，比 Vanilla 高 5.2 个百分点，且 on-topic 仅下降 4.3 个百分点。

### 跨架构泛化（Ministral-8B-Instruct）

| 方法 | Distractor | On-topic | Overall |
|------|-----------|---------|---------|
| Prompt Only | 0.25 | 0.98 | 0.62 |
| EnSToM @ layer 18 | 0.63 (+0.38) | 0.91 (-0.07) | **0.76** |

### 消融实验（阈值 $t$ 的影响）

| 阈值 $t$ | Distractor | On-topic | Overall |
|----------|-----------|---------|---------|
| Vanilla (固定) | 0.80 | 0.70 | 0.75 |
| $t = 2$ | 0.30 | 0.95 | 0.63 |
| $t = 7.5$ | 0.76 | 0.84 | **0.80** |
| $t = 9$ | ~baseline | 0.72 | ~0.6x |

### 数据效率

仅 10 个对比样本即可提取有效转向向量：distractor 准确率 0.74（vs 100 样本的 0.81），on-topic 0.85（vs 0.75），适合低资源场景。

### 关键发现

- **Layer 16 的熵分离最显著**：中间层编码语义信息，distractor 聚焦少量独特 token 导致低熵，on-topic 注意力分散导致高熵
- **跨领域一致性**：从银行、教育、健康、保险等不同领域提取的转向向量均有效，表明拒绝机制是通用的而非领域特定
- **任务泛化潜力**：在 jailbreak 防御任务中，Layer 33 的熵分布同样能区分有害和无害输入
- **系数分布分析**：82.5% 的 distractor 被分配 $C \geq 1.0$（强转向），45.8% 的 on-topic 被分配 $C < 0.5$（弱转向），符合设计预期
- **on-topic 对过度转向有一定鲁棒性**：即使 40.2% 的 on-topic 被分配 $C \geq 1.0$，准确率仍达 0.79

## 亮点与洞察

- 核心发现极其优雅：LLM 内部层级熵天然区分 on-topic/distractor，无需外部分类器
- 完全无训练的推理时干预，仅需 ~100 对比样本，部署成本极低
- 对层级功能分化的分析与认知科学发现一致：浅层捕捉句法、中层编码语义、深层整合上下文
- 动态系数比固定系数在"不伤害正常对话"方面优势明显

## 局限与展望

1. **层和阈值需手动选择**：熵提取层 $L$ 和阈值 $t$ 目前靠经验选择，未来需自动化
2. **硬负样本问题**：熵分布重叠区域的样本可能被错误分类，导致转向方向错误
3. **仅测试 7B/8B 模型**：未验证在 70B+ 大模型上的效果
4. **评估依赖 GPT-4o**：分类拒绝/回应的判断依赖 GPT-4o，可能引入偏差
5. **仅在银行领域深入评估**：跨领域实验仅用转向向量迁移，未做全面的领域适配

## 相关工作与启发

- **转向向量**：Turner et al. 2023 首提，Rimsky et al. 2024 应用于 LLaMA-2——本文增加熵缩放解决 on-topic 退化问题
- **主题维持**：CantTalkAboutThis (Sreedhar et al. 2024) 提供数据集，Llama Guard 通过指令微调实现安全守护——本文提供更轻量的替代
- **LLM 内部状态利用**：DoLa (Chuang et al. 2024) 用层间对比改善事实性，INSIDE (Chen et al. 2024) 用内部状态检测幻觉——本文用内部熵做输入分类
- **启发**：熵信号可扩展到更多场景（如检测幻觉、识别不确定性）；可与 LoRA 等参数高效方法结合

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 熵缩放转向向量的思路非常新颖，将 activation engineering 与内部信号巧妙结合
- **实验充分度**: ⭐⭐⭐⭐ — 多层分析、跨架构、跨领域、数据效率实验，但模型规模有限
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰、公式严谨、图表直观
- **价值**: ⭐⭐⭐⭐ — 为对话系统主题维持提供实用的零训练方案，对 activation engineering 领域有重要贡献

<!-- RELATED:START -->

## 相关论文

- [Dialogue Systems for Emotional Support via Value Reinforcement](dialogue_systems_for_emotional_support_via_value_reinforcement.md)
- [When Harry Meets Superman: The Role of The Interlocutor in Persona-Based Dialogue Generation](when_harry_meets_superman_the_role_of_the_interlocutor_in_persona-based_dialogue.md)
- [Wizard of Shopping: Target-Oriented E-commerce Dialogue Generation with Decision Tree Branching](wizard_of_shopping_target-oriented_e-commerce_dialogue_generation_with_decision_.md)
- [KokoroChat: A Japanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors](kokorochat_a_japanese_psychological_counseling_dialogue.md)
- [MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems](../../NeurIPS2025/dialogue/metamind_modeling_human_social_thoughts_with_metacognitive_multi-agent_systems.md)

<!-- RELATED:END -->
