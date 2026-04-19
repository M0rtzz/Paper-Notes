---
title: >-
  [论文解读] ChildMandarin: A Comprehensive Mandarin Speech Dataset for Young Children Aged 3-5
description: >-
   提出 ChildMandarin，一个面向 3-5 岁幼儿的普通话语音数据集，包含 397 名说话人、41.25 小时语音、覆盖中国 22 个省级行政区，并在 ASR 和说话人验证任务上提供了全面的基线评估。
tags:

---

# ChildMandarin: A Comprehensive Mandarin Speech Dataset for Young Children Aged 3-5

## 基本信息

- **会议**: ACL2025
- **arXiv**: [2409.18584](https://arxiv.org/abs/2409.18584)
- **代码**: [https://github.com/flageval-baai/ChildMandarin](https://github.com/flageval-baai/ChildMandarin)
- **领域**: Others (语音)
- **关键词**: 儿童语音识别, 普通话语音数据集, 3-5岁儿童, ASR, 说话人验证

## 一句话总结

提出 ChildMandarin，一个面向 3-5 岁幼儿的普通话语音数据集，包含 397 名说话人、41.25 小时语音、覆盖中国 22 个省级行政区，并在 ASR 和说话人验证任务上提供了全面的基线评估。

## 研究背景与动机

### 问题定义
自动语音识别（ASR）系统主要基于成人语音训练，在儿童语音上表现严重退化。幼儿语音具有发音不稳定、语调偏高、语速变化大、口齿不清等特点，与成人语音存在显著差异。特别是 3-5 岁幼儿的语音数据极度稀缺。

### 现有不足

**中文儿童语音数据集的现状**：
| 数据集 | 年龄 | 说话人数 | 时长 | 可用性 |
|---|---|---|---|---|
| Tong Corpus | 1;7-3;4 | 1 | 22h | 可用 |
| CASS CHILD | 1-4 | 23 | 631h | 不可用 |
| SLT-CSRC C1 | 7-11 | 927 | 28.6h | 不可用 |
| SLT-CSRC C2 | 4-11 | 54 | 29.5h | 不可用 |
| SingaKids | 7-12 | 255 | 75h | 可用 |

**Tong Corpus**：仅单一儿童纵向记录，无法提供 ASR 系统所需的说话人多样性

**CASS CHILD**：虽有 631 小时但仅 80 小时有转录且不公开

**SLT-CSRC**：仅在 SLT 2021 挑战赛期间开放

**SingaKids**：聚焦 7-12 岁，不覆盖幼儿年龄段

**英语数据集丰富但其他语言稀缺**：Providence、MyST、CSLU Kids 等英语资源较好，但非欧洲语言明显不足

### 动机
构建一个专门面向 3-5 岁幼儿的大规模、多说话人、地理分布广泛的普通话语音数据集，为该年龄段的 ASR 和说话人验证研究提供基础资源。

## 方法详解

### 整体框架
数据集构建 + ASR 基线评估 + 说话人验证基线评估

### 数据集构建

**采集设置**：
- 对话式录音（非朗读式），鼓励自然交互
- 父母全程在场提供情感支持
- 录音内容不受限制，聚焦适龄日常交流
- 设备：智能手机（Android 216 台、iPhone 181 台）
- 安静室内环境，容许少量背景噪声
- 格式：WAV PCM，16kHz 采样率，16-bit 精度

**标注**：
- 字符级手工转录，由专业转录员完成
- 忠实记录结巴、不流利和发育性语音模式
- 忠实记录地域发音变化
- 数字按实际发音转录

**数据规模**：
| 子集 | 说话人 | 语段数 | 时长(hrs) | 平均(s) |
|---|---|---|---|---|
| Train | 317 | 32,658 | 33.35 | 3.68 |
| Dev | 39 | 4,057 | 3.78 | 3.35 |
| Test | 41 | 4,198 | 4.12 | 3.53 |
| **总计** | **397** | **40,913** | **41.25** | **3.52** |

**人口统计**：
- 年龄分布：3/4/5 岁，各年龄性别均衡
- 地理分布：22 个省级行政区（山西 136 人最多，其次江苏 40、河南 39）
- 口音分类：轻度（多数）、中度、重度（约 4%）

### 关键设计

**说话人不重叠分割**：train/dev/test 间无说话人重叠，确保评估泛化能力。

**伦理保障**：
- 取得所有参与者父母/法定监护人的知情同意
- 每名儿童获得 150 元人民币公平补偿
- 数据匿名化，移除个人识别信息
- 仅限学术研究使用

## 实验

### ASR 任务

#### 从零训练模型

使用 Wenet 工具包训练，评估指标为 CER（%）：

| 模型 | 参数量 | 解码方式 | CER(%) |
|---|---|---|---|
| Transformer (CTC-AED) | 29M | Attention Rescoring | 32.15 |
| **Conformer (CTC-AED)** | **31M** | **Attention Rescoring** | **27.38** |
| Conformer (RNN-T AED) | 45M | Attention | 33.84 |
| Paraformer | 30M | Beam Search | 28.94 |

Conformer + CTC-AED + Attention Rescoring 表现最佳。

#### 自监督预训练模型微调

| 模型 | CER(%) |
|---|---|
| Wav2vec 2.0 (Base) | 20.29 |
| Wav2vec 2.0 (Large) | 21.12 |
| HuBERT (Base) | 18.74 |
| **HuBERT (Large)** | **14.97** |

HuBERT 一致优于 Wav2vec 2.0，与最新研究一致。

#### 监督预训练模型微调

| 模型 | 参数量 | Zero-shot | Fine-tuning |
|---|---|---|---|
| **CW (Conformer-WenetSpeech)** | **122M** | **18.05** | **13.66** |
| Whisper-tiny | 39M | 67.63 | 28.78 |
| Whisper-base | 74M | 51.49 | 23.33 |
| Whisper-small | 244M | 37.99 | 17.45 |
| Whisper-medium | 769M | 28.55 | 18.97 |
| Whisper-large-v2 | 1,550M | 29.43 | - |

- CW 在 zero-shot 和 fine-tuning 下均最优
- Fine-tuning 大幅降低所有模型的 CER
- Whisper-medium 微调后反而不如 Whisper-small（小数据集导致过拟合）

### 性能分析

**年龄和性别影响**：
- CER 随年龄增长而降低：3 岁 > 4 岁 > 5 岁
- 同年龄段男孩 CER 一致高于女孩
- 3 岁男孩 CER 最高（zero-shot 34.78%，fine-tuning 26.80%）

**错误类型分析**（CW 模型）：
| 年龄_性别 | 替换(%) | 删除(%) | 插入(%) |
|---|---|---|---|
| 3_F | 9.03 | 2.04 | 0.69 |
| 3_M | 26.80 | 4.35 | 2.11 |
| 4_F | 3.94 | 0.53 | 0.15 |
| 5_M | 14.32 | 3.04 | 1.23 |

替换错误占主导，其次是删除错误。

### 说话人验证任务

| 模型 | 参数量 | 嵌入维度 | PLDA EER(%) | PLDA minDCF |
|---|---|---|---|---|
| x-vector | 4.2M | 512 | 8.91 | 0.7198 |
| **ResNet-TDNN** | **15.5M** | **256** | **9.57** | **0.6597** |
| ECAPA-TDNN | 20.8M | 192 | 13.72 | 0.8697 |

- 数据集适合说话人验证任务
- ECAPA-TDNN 因参数量大在小数据集上过拟合，表现不如 x-vector 和 ResNet-TDNN
- 幼儿声道发育不完全导致性别相关特征被掩盖，增加了验证难度

### 关键发现
1. Fine-tuning 预训练模型比从零训练显著降低 CER（最佳从 27.38% 降至 13.66%）
2. 3 岁儿童语音是最大挑战，CER 可达同数据集 5 岁儿童的 2-3 倍
3. 对于小数据集，过大的模型反而可能因过拟合导致性能下降
4. Conformer-WenetSpeech 在儿童语音上的迁移效果最好

## 亮点与洞察

1. **填补关键年龄段空白**：专注 3-5 岁幼儿，这是现有资源最匮乏的年龄段
2. **高质量自然交互数据**：对话式采集优于朗读式，更真实地反映幼儿语音特点
3. **广泛地理覆盖**：22 个省级行政区 397 名说话人，具有良好的方言多样性
4. **双任务基线**：同时提供 ASR 和说话人验证基线，展示数据集的多任务适用性
5. **细粒度分析**：按年龄、性别、错误类型等多维度分析 ASR 性能，为后续研究提供方向

## 局限性

1. **数据量相对有限**：41.25 小时对比成人语音数据集仍然较小
2. **地理分布不均衡**：山西贡献最多说话人（136），部分省份仅有少量参与者
3. **大模型微调过拟合**：数据量限制了使用大参数模型的效果
4. **对话式录音的噪声控制**：幼儿录音中不可避免存在背景噪声

## 相关工作

- **中文儿童语音**：Tong Corpus (Xiangjun and Yip, 2017)、CASS CHILD (Gao et al., 2012)、SingaKids (Chen et al., 2016)、SLT-CSRC (Yu et al., 2021)
- **英文儿童语音**：MyST Corpus (Pradhan et al., 2024)、CSLU Kids (Shobaki et al., 2007)、TBALL (Kazemzadeh et al., 2005)
- **ASR 模型**：Conformer (Gulati et al., 2020)、Whisper (Radford et al., 2023)、HuBERT (Hsu et al., 2021)、Wav2vec 2.0 (Baevski et al., 2020)
- **说话人验证**：x-vector (Snyder et al., 2018)、ECAPA-TDNN (Desplanques et al., 2020)

## 评分 ⭐⭐⭐⭐

- 创新性：⭐⭐⭐⭐ — 填补 3-5 岁中文幼儿语音数据集空白
- 实用性：⭐⭐⭐⭐⭐ — 对教育技术和儿童人机交互有直接应用价值
- 方法新颖度：⭐⭐⭐ — 数据集构建为主，基线评估采用标准方法
- 实验充分度：⭐⭐⭐⭐ — ASR + SV 双任务，从零训练 + 预训练微调全面覆盖，细粒度分析详实

<!-- RELATED:START -->

## 相关论文

- [Towards Comprehensive Argument Analysis in Education: Dataset, Tasks, and Method](towards_comprehensive_argument_analysis_in_education_dataset_tasks_and_method.md)
- [QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning](qualispeech_a_speech_quality_assessment_dataset_with_natural_language_reasoning_.md)
- [PopAlign: Diversifying Contrasting Patterns for a More Comprehensive Alignment](popalign_diversifying_contrasting_patterns_for_a_more_comprehensive_alignment.md)
- [GA-S3: Comprehensive Social Network Simulation with Group Agents](ga-s3_comprehensive_social_network_simulation_with_group_agents.md)
- [Unlocking Speech Instruction Data Potential with Query Rewriting](unlocking_speech_instruction_data_potential_with_query_rewriting.md)

<!-- RELATED:END -->
