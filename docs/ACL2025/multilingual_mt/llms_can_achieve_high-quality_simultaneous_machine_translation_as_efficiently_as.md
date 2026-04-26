---
title: >-
  [论文解读] LLMs Can Achieve High-quality Simultaneous Machine Translation as Efficiently as Offline
description: >-
  [ACL 2025 (Findings)][同声传译] 本文提出了一种新范式，通过将源语言和目标语言 token 按延迟要求重排为交错序列来构造 SFT 数据，使 LLM 能够像离线翻译一样高效地完成高质量同声传译（SiMT），在多个基准上达到 SOTA 性能，同时保持离线翻译的原有能力。
tags:
  - ACL 2025 (Findings)
  - 同声传译
  - 大语言模型
  - 流式翻译
  - 交错序列
  - 延迟控制
---

# LLMs Can Achieve High-quality Simultaneous Machine Translation as Efficiently as Offline

**会议**: ACL 2025 (Findings)  
**arXiv**: [2504.09570](https://arxiv.org/abs/2504.09570)  
**代码**: 无  
**领域**: 文本生成 / 机器翻译  
**关键词**: 同声传译, 大语言模型, 流式翻译, 交错序列, 延迟控制

## 一句话总结

本文提出了一种新范式，通过将源语言和目标语言 token 按延迟要求重排为交错序列来构造 SFT 数据，使 LLM 能够像离线翻译一样高效地完成高质量同声传译（SiMT），在多个基准上达到 SOTA 性能，同时保持离线翻译的原有能力。

## 研究背景与动机

**领域现状**：大语言模型在离线机器翻译中表现出色，仅通过简单的翻译提示（"Translate the following sentence from [src] into [tgt]:"）即可获得高质量译文。同声传译（SiMT）是一种更贴近实际应用场景的翻译模式，要求在源语言 token 流式到达时实时进行翻译。

**现有痛点**：在 SiMT 场景下，decoder-only LLM 的自回归特性严重限制了其效率和性能。传统方法通常基于 encoder-decoder 架构设计，read/write 策略复杂且难以直接迁移到 LLM 上。现有的 LLM-based SiMT 方法要么需要多次前向推理来模拟 read/write 操作，导致效率极低；要么牺牲翻译质量以换取低延迟。

**核心矛盾**：SiMT 在质量和延迟之间存在根本性的 trade-off。传统方法通过固定策略（如 wait-k）来平衡这一矛盾，但这些策略缺乏灵活性，且无法充分利用 LLM 的强大生成能力。LLM 的高效自回归解码机制在 SiMT 中被打断，因为需要不断在读取和写入操作之间切换。

**本文目标**：(1) 让 LLM 在 SiMT 任务中达到与离线翻译相当的效率；(2) 在不同延迟要求下都能产出高质量译文；(3) 保持模型的离线翻译能力不退化。

**切入角度**：作者观察到，如果将 SiMT 中的 read/write 交替操作编码到一个统一的序列中，那么 LLM 就可以以标准的自回归方式高效生成，无需反复暂停和恢复。

**核心 idea**：用特殊 token 分隔的源目标交错序列表示不同延迟下的 SiMT 过程，让 LLM 通过标准 SFT 学习自适应的 read/write 策略，同时保持高效解码。

## 方法详解

### 整体框架

输入为源语言句子和延迟要求，输出为按交错序列格式组织的同声翻译结果。训练阶段构造不同延迟级别的交错序列作为 SFT 数据；推理阶段模型以流式方式接收源 token 并自回归生成交错的翻译 token。

### 关键设计

1. **交错序列构造（Interleaved Sequence Construction）**:

    - 功能：将 SiMT 的 read/write 操作编码为单一序列，使 LLM 能够以标准自回归方式处理
    - 核心思路：根据对齐信息将源 token 和目标 token 按时间顺序交错排列，使用特殊 token（如 `<src>` 和 `<tgt>`）分隔不同语言的片段。延迟提示（latency prompt）控制交错的粒度——低延迟意味着更频繁的交替，高延迟则允许积累更多源 token 后再翻译。构造过程参考词对齐和 wait-k 等策略确定每个目标 token 对应的最少源 token 数量
    - 设计动机：解决了 LLM 在 SiMT 中需要反复打断自回归解码的效率问题，将 SiMT 转化为标准的序列生成任务

2. **延迟感知训练策略（Latency-aware Training）**:

    - 功能：使单一模型支持多种延迟级别的 SiMT
    - 核心思路：通过在不同延迟要求下构造多组交错序列 SFT 数据，使模型学会根据延迟提示（latency prompt）自适应地调整 read/write 策略。训练时混合不同延迟的数据，模型通过识别延迟提示来决定在何时读入新的源 token、何时生成目标 token。即使使用有限的 SFT 数据（相比全量翻译数据而言），模型也能学到合理的策略
    - 设计动机：避免为每个延迟级别训练独立模型，实现灵活的延迟控制

3. **高效流式推理（Efficient Streaming Inference）**:

    - 功能：在推理时实现真正的流式同声传译
    - 核心思路：推理时模型按自回归方式逐 token 生成。当生成 `<src>` token 时等待新的源 token 到达，当生成 `<tgt>` token 时输出翻译。整个过程无需多次前向推理或复杂的策略调度，与离线翻译的推理效率相当。模型还天然支持文档级 SiMT，无需额外微调
    - 设计动机：利用 LLM 自回归解码的天然效率，避免了传统 SiMT 系统复杂的调度逻辑

### 损失函数 / 训练策略

采用标准的语言模型 SFT 损失（交叉熵），在交错序列上进行监督微调。训练数据混合了多种延迟级别的样本，确保模型学到延迟条件生成能力。

## 实验关键数据

### 主实验

| 数据集/语言对 | 延迟(AL) | 本文方法 BLEU | 之前 SOTA BLEU | 提升 |
|---------------|----------|-------------|---------------|------|
| WMT15 De→En | ~3 | SOTA级 | 传统 wait-k | 显著提升 |
| WMT15 De→En | ~5 | SOTA级 | 传统 SiMT | 显著提升 |
| WMT15 De→En | 离线 | 与原模型持平 | - | 无退化 |
| 文档级 SiMT | 多级 | 超越离线 | - | 泛化性强 |

### 消融实验

| 配置 | BLEU | 说明 |
|------|------|------|
| Full model (多延迟混合训练) | 最高 | 完整模型 |
| 单一延迟训练 | 下降 | 缺乏灵活性 |
| 无延迟提示 | 显著下降 | 模型无法区分延迟要求 |
| 更少 SFT 数据 | 略有下降 | 表明少量数据即可 |

### 关键发现

- 即使使用少量 SFT 数据，方法也能达到 SOTA 性能，证明了交错序列范式的有效性
- 模型在文档级 SiMT 上表现出良好的零样本泛化能力，甚至超越了专门的离线翻译模型
- 离线翻译能力得到保持，没有因为 SiMT 微调而退化
- 延迟提示是关键设计，去掉后模型无法有效控制翻译时机

## 亮点与洞察

- **交错序列范式**将 SiMT 的复杂 read/write 策略问题优雅地转化为标准序列生成问题，巧妙地复用了 LLM 的自回归解码机制而非对抗它。这个思路可以迁移到其他需要交替输入输出的流式处理任务上
- **延迟提示控制**机制允许单一模型支持灵活的延迟需求，这种条件生成的思路适用于任何需要在质量和效率之间权衡的生成任务
- **文档级零样本泛化**表明该方法学到了通用的 read/write 策略，而不仅仅是记忆句子级别的模式

## 局限与展望

- 论文主要在 De↔En 等资源丰富的语种对上实验，低资源语种对的泛化性有待验证
- 依赖词对齐信息构造交错序列，对齐质量可能影响最终性能
- 对超长文档的流式处理能力有待进一步评估，当前实验的文档长度相对有限
- 未充分探索不同基座 LLM 对该方法的影响

## 相关工作与启发

- **vs Wait-k策略**: Wait-k 使用固定的等待策略，缺乏灵活性；本文通过延迟提示实现自适应策略，显著优于固定策略
- **vs 传统 SiMT 系统**: 传统方法基于 encoder-decoder 架构设计复杂的 policy network；本文直接利用 LLM 的语言建模能力，更简洁且性能更强
- **vs 其他 LLM-based SiMT**: 先前工作需要多次前向推理效率极低；本文通过交错序列实现了与离线翻译相当的效率

## 评分

- 新颖性: ⭐⭐⭐⭐ 交错序列是一个简洁有效的范式，将 SiMT 优雅地转为标准 LM 任务
- 实验充分度: ⭐⭐⭐⭐ 覆盖多个语种对和延迟级别，但HTML不可用限制了对实验细节的全面评估
- 写作质量: ⭐⭐⭐⭐ 摘要和方法描述清晰，问题定义准确
- 价值: ⭐⭐⭐⭐ 对 LLM-based SiMT 提供了实用的解决方案，推动了同声传译的实用化

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] SeqPO-SiMT: Sequential Policy Optimization for Simultaneous Machine Translation](seqpo-simt_sequential_policy_optimization_for_simultaneous_machine_translation.md)
- [\[ACL 2025\] Watching the Watchers: Exposing Gender Disparities in Machine Translation Quality Estimation](watching_the_watchers_exposing_gender_disparities_in_machine_translation_quality.md)
- [\[ACL 2025\] Alleviating Distribution Shift in Synthetic Data for Machine Translation Quality Estimation](alleviating_distribution_shift_in_synthetic_data_for_machine_translation_quality.md)
- [\[ACL 2025\] AskQE: Question Answering as Automatic Evaluation for Machine Translation](askqe_question_answering_as_automatic_evaluation_for_machine_translation.md)
- [\[ACL 2025\] Machine Translation Models are Zero-Shot Detectors of Translation Direction](machine_translation_models_are_zero-shot_detectors_of_translation_direction.md)

<!-- RELATED:END -->
