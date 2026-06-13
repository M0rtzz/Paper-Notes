---
title: >-
  [论文解读] ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging
description: >-
  [ACL 2025][LLM安全][机器遗忘] 在 SemEval-2025 Task 4（LLM 敏感内容遗忘）中获得第二名，核心思路是训练两个互补模型（一个过度遗忘、一个遗忘不足），通过 TIES-Merging 合并得到平衡遗忘的模型，本地实验达到近乎完美的 MIA 分数 0.501。 机器遗忘（Machine Unl…
tags:
  - "ACL 2025"
  - "LLM安全"
  - "机器遗忘"
  - "模型合并"
  - "TIES-Merging"
  - "隐私保护"
---

# ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging

**会议**: ACL 2025  
**arXiv**: [2503.21088](https://arxiv.org/abs/2503.21088)  
**代码**: 有 ([https://github.com/zjunlp/unlearn/tree/main/semeval25](https://github.com/zjunlp/unlearn/tree/main/semeval25))  
**领域**: NLP / 机器遗忘  
**关键词**: 机器遗忘, 模型合并, TIES-Merging, 隐私保护, LLM安全

## 一句话总结

在 SemEval-2025 Task 4（LLM 敏感内容遗忘）中获得第二名，核心思路是训练两个互补模型（一个过度遗忘、一个遗忘不足），通过 TIES-Merging 合并得到平衡遗忘的模型，本地实验达到近乎完美的 MIA 分数 0.501。

## 研究背景与动机

机器遗忘（Machine Unlearning）是 AI 安全领域的关键技术，旨在从训练好的模型中选择性地移除敏感数据（如版权材料和个人信息）。然而，现有遗忘方法面临根本性困难：

**过度遗忘（Over-forgetting）**：移除了过多的非敏感信息，导致模型性能综合下降

**遗忘不足（Under-forgetting）**：未能完全移除目标敏感数据

**超参数敏感性**：找到能在多个评估维度上平衡性能的最优超参数极其困难，有时甚至不可能

作者的关键洞察是：与其试图找到一组完美的超参数，不如训练两个各有偏向的模型，然后通过模型合并来综合它们的优势。

## 方法详解

### 整体框架

两阶段系统：训练阶段 → 合并阶段

### 关键设计

1. **训练阶段——两个互补模型**：
   使用 LoRA（低秩适应）微调 OLMo-7B-0724-Instruct，训练目标相同但超参数不同，分别产生：
    - **model₁（过度遗忘）**：Task Aggregate 高（0.968），but MIA Score 低（0.022）——遗忘太多，连不该忘的也忘了
    - **model₂（遗忘不足）**：Task Aggregate 低（0.659），but MIA Score 高（0.818）——保留了太多应该遗忘的信息

   训练损失函数包含三个组件：
    - **NPO（负偏好优化）**：在遗忘集上最小化目标 token 的概率
    - **GDR（保留集梯度下降）**：在保留集上保持模型原有能力
    - **KLR（保留集 KL 散度最小化）**：确保遗忘后模型在保留集上的输出分布不偏移

   总损失：L_total = α·L_npo + β·L_gdr + γ·L_klr

2. **合并阶段——TIES-Merging**：
   对两个模型的 LoRA 适配器进行三步合并：
    - **Trimming（修剪）**：按密度阈值保留最重要的参数，将其余置零。密度 0.8 取得最佳效果——低密度过度裁剪，高密度引入冗余
    - **Electing（选举）**：基于参数绝对值创建统一符号向量，解决参数方向冲突
    - **Disjoint Merging（析取合并）**：仅对符号一致的非零参数取平均，丢弃冲突参数

### 损失函数 / 训练策略

- NPO 损失：类似 DPO 但只针对遗忘集，逆向优化使模型降低对遗忘数据的生成概率
- GDR 损失：标准交叉熵，在保留集上维持模型能力
- KLR 损失：遗忘后模型与原始模型在保留集上的 KL 散度最小化
- 两个模型通过不同的 α、β、γ 配比实现互补偏向

## 实验关键数据

### 在线和本地实验结果

| 环境 | 方法 | Aggregate | Task Aggregate | MIA Score/AUC | MMLU Avg. |
|------|------|-----------|---------------|---------------|-----------|
| Online | AILS-NTUA (第一) | 0.706 | 0.827 | 0.847/– | 0.443 |
| Online | **ZJUKLAB (ours, 第二)** | **0.487** | **0.944** | 0.048/– | 0.471 |
| Local | model₁ (过度遗忘) | 0.481 | 0.968 | 0.045/0.022♣ | 0.431 |
| Local | model₂ (不足遗忘) | 0.504 | 0.659 | 0.364/0.818♠ | 0.491 |
| Local | **Merged (ours)** | **0.806** | **0.939** | **0.997/0.501♡** | **0.480** |

### 合并方法对比

| 合并方法 | Aggregate |
|---------|-----------|
| Linear | 0.244 |
| DARE-Linear | 0.440 |
| DARE-TIES | 0.561 |
| Magnitude Prune | 0.558 |
| **TIES** | **0.806** |

### 密度参数消融

| 密度 | 效果 |
|------|------|
| 0.6 | 过度裁剪，MIA 过低 |
| **0.8** | **最优平衡** |
| 1.0 | 引入冗余，MIA 偏高 |

### 关键发现

1. **模型合并思路非常有效**：本地实验中合并模型的 MIA AUC 达到 0.501（几乎完美），Aggregate 从两个子模型的 ~0.49 跃升至 0.806
2. **TIES-Merging 大幅优于其他合并方法**：比 Linear 高出 0.562，比 DARE-TIES 高出 0.245
3. **训练过程分析**：Regurgitation Score 和 Knowledge Score 在训练早期（epochs 0-0.8）同时下降，之后 Knowledge Score 稳步上升而 Regurgitation Score 波动上升，反映了遗忘与保留的优化方向逐渐分化
4. **参数变化方向分析**：训练拐点前后的参数变化向量夹角约 70-85 度，说明优化方向发生了显著转变

## 亮点与洞察

- **"两个有缺陷的模型合并成一个好模型"** 的思路简单但深刻——避免了在单一模型中寻找完美平衡的困难
- 详细的训练过程分析（性能轨迹、损失动力学、参数角度分析）提供了对遗忘过程的深入理解
- 对遗忘评估指标的批判性分析很有价值：ROUGE 度量文本重叠而非知识泄露，MIA 也存在可靠性问题
- 指出了过度遗忘导致模型崩塌（生成 "6 6 6" 重复输出）的现象，是对实际部署的重要警告

## 局限与展望

1. **过度遗忘现象未完全解决**：合并后模型仍可能产生退化输出
2. **通用知识遗忘**：遗忘操作导致 "法国首都是什么" 等无关知识的准确率从 0.88 降至 0.35
3. **评估指标的可靠性**：ROUGE-L 无法检测语义级信息泄露，MIA 分数的可靠性也受质疑
4. **线上线下差异**：线上 MIA Score (0.048) 与本地 (0.997) 差异巨大，可能因数据集差异
5. 当前范式缺乏正向引导（如强化学习），模型只能在反向优化压力下退化

## 相关工作与启发

- NPO (Zhang et al., 2024) 是核心遗忘算法，本文通过模型合并克服了其单一使用时的过/欠遗忘问题
- TIES-Merging (Yadav et al., 2023) 原用于多任务学习的模型合并，被巧妙迁移到遗忘场景
- 国际 AI 安全报告 (Bengio et al., 2025) 指出当前遗忘方法仍不充分，与本文分析一致
- 未来方向：结合数据增强和强化学习引入正向信号，探索按需遗忘

## 评分

- **新颖性**: ⭐⭐⭐⭐ "过度+不足→合并平衡" 的思路简洁优雅
- **实验充分度**: ⭐⭐⭐⭐⭐ 在线比赛验证 + 本地详细分析 + 多维度消融 + 合并方法对比
- **写作质量**: ⭐⭐⭐⭐ 分析深入透彻，尤其是 Sec. 5-6 的分析和反思
- **价值**: ⭐⭐⭐⭐ 对机器遗忘领域的方法论和评估都有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs](lacuna_inc_at_semeval-2025_task_4_lora-enhanced_influence-based_unlearning_for_l.md)
- [\[ACL 2025\] CLIPErase: Efficient Unlearning of Visual-Textual Associations in CLIP](cliperase_efficient_unlearning_of_visual-textual_associations_in_clip.md)
- [\[ICML 2025\] NegMerge: Sign-Consensual Weight Merging for Machine Unlearning](../../ICML2025/llm_safety/negmerge_sign-consensual_weight_merging_for_machine_unlearning.md)
- [\[ACL 2025\] Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models](merge_hijacking_backdoor_attacks_to_model_merging_of_large_language_models.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
