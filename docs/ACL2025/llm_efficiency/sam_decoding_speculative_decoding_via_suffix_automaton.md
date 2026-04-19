---
title: >-
  [论文解读] SAM Decoding: Speculative Decoding via Suffix Automaton
description: >-
  [ACL 2025][LLM效率][推测解码] 提出SAM-Decoding，利用后缀自动机（Suffix Automaton）对通用文本语料和当前文本序列进行最长后缀匹配来高效生成推测解码的草稿，平均O(1)时间复杂度，在Spec-Bench上比现有检索式方法快18%+，并可与EAGLE-2等方法互补组合进一步提速3.28%-11.13%。
tags:
  - ACL 2025
  - LLM效率
  - 推测解码
  - 后缀自动机
  - 无损加速
  - LLM推理优化
  - 检索式草稿生成
---

# SAM Decoding: Speculative Decoding via Suffix Automaton

**会议**: ACL 2025  
**arXiv**: [2411.10666](https://arxiv.org/abs/2411.10666)  
**代码**: [GitHub仓库](https://github.com/hyx1999/SAM-Decoding)  
**领域**: LLM效率  
**关键词**: 推测解码, 后缀自动机, 无损加速, LLM推理优化, 检索式草稿生成

## 一句话总结
提出SAM-Decoding，利用后缀自动机（Suffix Automaton）对通用文本语料和当前文本序列进行最长后缀匹配来高效生成推测解码的草稿，平均O(1)时间复杂度，在Spec-Bench上比现有检索式方法快18%+，并可与EAGLE-2等方法互补组合进一步提速3.28%-11.13%。

## 研究背景与动机
- **领域现状**：推测解码（Speculative Decoding）是LLM无损推理加速的有效技术，通过轻量级草稿模型快速生成候选token，再由目标LLM并行验证。方法分为模型式（如Medusa、EAGLE-2）和无模型式（检索式，如PLD、REST）
- **现有痛点**：
    - 检索源单一：PLD仅用当前文本，REST仅用外部语料
    - 检索效率不足：PLD暴力搜索n-gram匹配复杂度O(n²L)，REST用后缀数组复杂度O(n²logL)
    - 领域局限性：检索式方法主要在摘要和RAG等特定领域有效，其他领域加速有限
- **核心矛盾**：检索式方法的理论优势（无需训练额外模型、可生成长草稿）受限于检索效率和适用范围
- **切入角度**：利用后缀自动机这一经典数据结构，同时解决检索效率和资源覆盖问题
- **核心idea**：后缀自动机天然支持O(1)平均时间的最长后缀匹配，且能精确找到匹配位置和长度，远优于n-gram匹配

## 方法详解

### 整体框架
SAM-Decoding构建两种后缀自动机，在每轮生成时通过自动机检索草稿，结合辅助SD方法自适应选择最优草稿，经LLM验证后更新自动机状态。Pipeline：
1. 离线构建静态SAM（文本语料）+ 在线构建动态SAM（当前文本）
2. 每步生成：双SAM并行检索 → 可选辅助方法生成草稿 → 根据匹配长度自适应选择 → LLM验证 → 更新SAM状态

### 关键设计
1. **静态后缀自动机（Static SAM）**:

    - 离线对通用文本语料（Stanford-alpaca、python-code-instruction-18k、GSM8k的Vicuna-7B生成结果）构建
    - 将多个字符串用EOS拼接后构建单一SAM
    - 每个节点记录 `min_endpos`（对应子串的最早结束位置），用于定位匹配后缀
    - 额外预计算 `topk_succ`（top-k后继状态），用Prim算法构建树状草稿

2. **动态后缀自动机（Dynamic SAM）**:

    - 对当前生成的文本序列（包括用户提示和已生成token）在线增量构建
    - 每次验证接受新token后，先进行状态转移再扩展自动机结构
    - 动态SAM的草稿通常优于静态SAM，因此默认优先使用

3. **O(1)平均时间的状态转移**:

    - 利用SAM的两种边（extension edge + suffix link edge），在生成过程中追踪当前最长后缀匹配
    - 通过摊还分析证明平均O(1)时间复杂度，最坏O(L)
    - 对比：PLD为O(n²L)，REST为O(n²logL)
    - 且无需预设匹配长度n的上限

4. **自适应草稿选择**:

    - 核心洞察：后缀匹配长度 $l$ 可指示草稿质量，长匹配意味着更多token可能被接受
    - 设定阈值 $l_{\text{threshold}}$：若SAM匹配长度 > 阈值，用SAM草稿；否则用辅助方法
    - 静态vs动态选择：仅当 $l_1 > l_2 + l_{\text{bias}}$ 时用静态SAM（动态优先）
    - 可与Token Recycling（无模型）和EAGLE-2（有模型）灵活组合

### 损失函数 / 训练策略
- SAM-Decoding本身**无需训练**，纯算法方法
- 与EAGLE-2组合时使用其原有训练好的draft模型
- 超参数设置：$l_{\text{bias}} = l_{\text{threshold}} = 5$，草稿大小默认40（代码数据集为16）

## 实验关键数据

### 主实验
| 数据集 | 指标 | SAM-Decoding | PLD (SOTA检索) | SAM+EAGLE-2 | 纯EAGLE-2 |
|--------|------|------|----------|------|------|
| Spec-Bench | Speedup | **1.84×** | 1.56× | **2.58×** | 2.38× |
| HumanEval | Speedup | **2.29×** | 1.52× | **3.35×** | 3.24× |
| HAGRID | Speedup | **2.24×** | 1.29× | **2.81×** | 2.41× |
| Spec-Bench | MAT | 2.30 | 1.75 | 4.62 | 4.36 |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| w/o Static SAM | 1.64× | 静态SAM贡献0.20×加速 |
| w/o Dynamic SAM | 1.33× | 动态SAM贡献更大（0.51×），从上下文生成草稿更有效 |
| $l_{\text{bias}}$/$l_{\text{threshold}}$ grid search | MAT/Speedup | 两者=5时最优，超过5后性能下降 |
| 草稿大小 | Throughput | 40最优，超过40后GPU压力增大导致减速 |

### 关键发现
- 动态SAM（当前文本匹配）的贡献远大于静态SAM（语料匹配），说明从自身上下文生成草稿往往更有效
- SAM-Decoding在多轮对话、摘要、RAG任务上加速最显著（这些任务自然包含大量重复/引用）
- 与EAGLE-2组合后，在Vicuna-7B/13B/33B上都有一致提升，说明方法的模型规模通用性
- 在HumanEval（代码生成）上，与EAGLE-2组合时提升有限，因代码生成较少复制已生成文本

## 亮点与洞察
- 将后缀自动机引入推测解码领域，是数据结构与LLM推理优化的优雅结合
- O(1)平均时间+精确最长匹配+无需预设匹配长度，三重优势远胜现有n-gram方法
- 自适应选择策略使方法可作为任何现有SD方法的"插件"，具有很强的实用价值
- 静态SAM离线构建、动态SAM在线增量更新的设计兼顾了效率和灵活性

## 局限与展望
- 静态语料来源于特定LLM（Vicuna-7B）的生成结果，跨模型迁移性存疑
- 静态语料覆盖不够多样，限制了在未见过的领域的性能
- 自适应选择策略过于启发式（简单比较匹配长度），未充分利用匹配信息
- **可探索方向**：(1) 训练一个分类器来选择不同解码策略；(2) 构建更多样化的语料库针对不同任务；(3) 将匹配长度信号融入草稿树的构建过程中

## 相关工作与启发
- PLD (Prompt Lookup Decoding) 开创了从当前文本中检索n-gram匹配作为草稿的思路
- REST利用后缀数组从外部语料检索草稿，SAM-Decoding在数据结构选择上更优
- EAGLE-2通过浅层Transformer预测hidden states生成草稿，与SAM是互补关系
- Token Recycling利用已生成的token分布构建草稿树，与SAM组合效果显著

## 评分
- 新颖性: ⭐⭐⭐⭐ 后缀自动机应用于推测解码是明确创新，但整体仍是检索式SD的改进
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多模型规模、多种组合方式、详细消融、复杂度分析，非常充分
- 写作质量: ⭐⭐⭐⭐ 算法描述清晰，理论分析严谨，图示直观
- 价值: ⭐⭐⭐⭐⭐ 作为免训练方法可直接集成到现有SD框架中，实用价值极高

<!-- RELATED:START -->

## 相关论文

- [3-Model Speculative Decoding (PyramidSD)](../../NeurIPS2025/llm_efficiency/3model_speculative_decoding.md)
- [Tetris: Optimal Draft Token Selection for Batch Speculative Decoding](tetris_optimal_draft_token_selection_for_batch_speculative_decoding.md)
- [CLaSp: In-Context Layer Skip for Self-Speculative Decoding](clasp_self_speculative_decoding.md)
- [OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding](../../NeurIPS2025/llm_efficiency/omnidraft_a_cross-vocabulary_online_adaptive_drafter_for_on-device_speculative_d.md)
- [Mitigating Posterior Salience Attenuation in Long-Context LLMs with Positional Contrastive Decoding](mitigating_posterior_salience_attenuation_in_long-context_llms_with_positional_c.md)

<!-- RELATED:END -->
