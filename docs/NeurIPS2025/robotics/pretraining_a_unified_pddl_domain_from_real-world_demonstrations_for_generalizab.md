---
title: >-
  [论文解读] UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Task Planning
description: >-
  [NeurIPS 2025][机器人][PDDL] UniDomain 从 12,393 个真实机器人操作视频中预训练统一的 PDDL 规划域（含 3,137 个算子和 2,875 个谓词），通过层级融合构建元域，实现零样本跨任务符号规划，比最强基线高出 58% 成功率和 160% 计划最优性。
tags:
  - NeurIPS 2025
  - 机器人
  - PDDL
  - 任务规划
  - 机器人操作
  - 知识蒸馏
  - 大规模演示学习
---

# UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Task Planning

**会议**: NeurIPS 2025  
**arXiv**: [2507.21545](https://arxiv.org/abs/2507.21545)  
**代码**: https://roboticsjtu.github.io/UniDomain/  
**领域**: 机器人 / 任务规划  
**关键词**: PDDL, 任务规划, 机器人操作, 知识蒸馏, 大规模演示学习

## 一句话总结

UniDomain 从 12,393 个真实机器人操作视频中预训练统一的 PDDL 规划域（含 3,137 个算子和 2,875 个谓词），通过层级融合构建元域，实现零样本跨任务符号规划，比最强基线高出 58% 成功率和 160% 计划最优性。

## 研究背景与动机

机器人任务规划需要在自然语言指令和视觉观测中推理隐含约束。例如，"将积木按奇偶分组并升序排列"隐含了拆栈、排序、放置的长程依赖；"泡杯茶"需要开柜子、找茶杯、烧水等一系列前置步骤。这些任务需要对动作前提条件、时序依赖和物理约束进行结构化推理。

**现有方法的局限**：
- LLM/VLM 直接规划（如 Code-as-Policies、ReAct）：强大的常识先验，但无法准确建模动作前后条件，长程规划容易出错
- LLM + PDDL 混合方法（如 LLM+P、ISR-LLM）：依赖手工制作的 PDDL 域或 LLM 从语言直接生成域，域质量有限
- 从演示学习 PDDL 域：已有工作仅从单个或少量演示学习窄域，需要任务特定先验

**核心矛盾**：高质量的 PDDL 域是符号规划的关键，但手工制作代价高且泛化受限；LLM 直接生成的域质量不足；从少量演示学习的域覆盖范围太窄。

**本文切入角度**：借鉴基础模型的"预训练-后训练-推理"范式，从大规模机器人操作数据集（DROID）中预训练通用 PDDL 域，然后通过域融合进行"后训练"以适配特定任务类别。

## 方法详解

### 整体框架

UniDomain 包含三个阶段：
1. **域预训练**（Domain Pretraining）：从视频演示中提取原子域，构建统一域
2. **域融合**（Domain Fusion）：检索相关原子域，层级融合为元域
3. **在线规划**（Online Planning）：用元域构建 PDDL 问题并求解

### 关键设计

1. **基于能量的关键帧提取**：提出了一种简单高效的无域关键帧提取方法。计算灰度帧的像素能量 $E(I_t) = \sum_{i,j} I_t(i,j)^2$，通过滑动窗口检测能量序列的局部极值来选取关键帧。相比基于 CLIP/SigLIP 嵌入的方法，处理速度从 47.8 秒/视频降至 0.6 秒/视频，且准确率更高（28% vs 15% 的单次成功率）。

2. **闭环原子域生成**：给定关键帧序列和任务指令，VLM 推断每个帧转换的算子（前提条件和效果），LLM 进行整体修订确保语法正确和谓词一致性。然后进行两层嵌套验证：

    - **可解性检查**：LLM 生成 K=5 个测试问题，PDDL求解器检查是否可解，可解性分数 $S(D_r) = \frac{1}{K}\sum_k \mathbb{I}[\text{solver solves } P_k]$，阈值 θ=0.6
    - **解验证**：另一个 LLM 检查最难测试问题的解是否符合物理和常识约束
   两层检查迭代最多 L=5 次。

3. **层级二叉树域融合**：将检索到的原子域沿二叉树递归合并。每次融合两步：

    - **谓词合并**：计算语义嵌入的余弦相似度（阈值 τ_p=0.3），LLM 验证语义等价性后合并
    - **算子合并**：类似地计算名称嵌入相似度（阈值 τ_o=0.3），合并功能等价的算子，继承前提条件和效果的并集
   
   层级融合避免了直接 LLM 合并的结构错误问题。

4. **任务相关过滤**：在线规划时，先用 VLM 生成初始 PDDL 问题提取相关谓词集 P₀，再根据 P₀ 从元域中提取相关算子 O' = O_pre ∪ O_eff，构建紧凑域 D_new 进行规划。

### 损失函数 / 训练策略

本文不涉及传统神经网络训练。核心的"训练"是通过 LLM/VLM 的闭环验证来迭代优化 PDDL 域的质量。评估指标包括：可解性分数 S(D)、方案验证通过率、以及下游任务成功率。

## 实验关键数据

### 主实验

| 方法 | 类型 | 成功率(SR)↑ | SPL↑ | OR(K=0)↑ |
|------|------|-----------|------|----------|
| Code-as-Policies | LLM直接 | 51% | - | 低 |
| ReAct | LLM+反馈 | 较高 | - | 低 |
| VLM-CoT | VLM | 中等 | - | 中等 |
| ISR-LLM | LLM+PDDL | 最高（基线中） | - | 低 |
| BoN-iVML | LLM+PDDL | 中等 | - | 中等 |
| **UniDomain** | **预训练域+PDDL** | **85%** | **最高** | **83%** |

UniDomain 比最强基线高 58% 成功率，160% 计划最优性提升。83% 的任务中产生了最优计划。

### 消融实验

| 消融配置 | 成功率 | 关键观察 |
|---------|--------|---------|
| 完整 UniDomain | 85% | 最佳性能 |
| w/o 闭环验证 (w/o CL) | 显著下降 | 单次 LLM 生成域质量差 |
| w/o 域融合 | 19% | 单个原子域无法组合泛化 |
| w/o 结构化融合 | 0% (语法错误) | 直接 LLM 合并完全失败 |
| w/o 谓词分组 | 下降（尤其组合域） | LLM 难以理解扁平谓词列表 |
| w/o 算子过滤 | 下降（尤其积木域） | 无关符号干扰长程推理 |

### 关键发现

- 预训练统一域 + 域融合 + 在线过滤的三阶段流水线至关重要，每个阶段移除都导致显著性能下降
- 基于能量的关键帧提取比基于视觉模型的方法快 80 倍且更准确
- 直接 LLM 合并多个域会产生结构错误，层级融合是必要的
- 83% 的任务通过组合已学算子产生最优计划，展现了强大的组合泛化能力
- UniDomain 的 LLM 调用次数和思考时间均为最优方法中最低

## 亮点与洞察

- **领域首创**：首个从大规模真实演示预训练通用 PDDL 域的框架，类比 LLM 的预训练范式
- **设计巧妙的闭环验证**：用 PDDL 求解器本身作为域质量的验证器，无需人类反馈
- **组合泛化**：通过连接独立的操作行为（如 pick、pour、stir），解决复合长程任务
- **实用性强**：生成的计划可直接转化为自然语言指令输入 VLA 模型执行

## 局限与展望

- 自动检索的原子域可能冗余，元域构建耗时
- 仅支持 PDDL 1.0，缺少时序约束、数值流和代价敏感规划
- 实验假设完全可观测，未处理遮挡和感知噪声
- 评估使用人类遥操作作为低级控制，未端到端验证完整机器人系统
- 关键帧提取基于简单像素能量，可能遗漏语义上重要的细微变化

## 相关工作与启发

UniDomain 巧妙地将 LLM 时代的预训练-后训练-推理范式引入符号规划领域。与 ISR-LLM 和 NL2Plan 等从语言直接生成域的方法不同，UniDomain 从视觉演示中获取接地的操作知识。与 BLADE 等从单一演示学习窄域的方法不同，UniDomain 从大规模数据中学习覆盖广泛任务空间的统一域。域融合方法类似于知识图谱合并，但专门针对 PDDL 的结构化特性设计。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首创从大规模演示预训练 PDDL 域的范式，视角独到
- 实验充分度: ⭐⭐⭐⭐ 100个真实任务评估+充分消融，但缺少端到端机器人验证
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，三阶段结构易于理解
- 价值: ⭐⭐⭐⭐⭐ 为机器人任务规划提供了可扩展的新范式，实际意义重大

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)
- [\[ICCV 2025\] Bridging Domain Generalization to Multimodal Domain Generalization via Unified Representations](../../ICCV2025/robotics/bridging_domain_generalization_to_multimodal_domain_generalization_via_unified_r.md)
- [\[NeurIPS 2025\] Towards Reliable Code-as-Policies: A Neuro-Symbolic Framework for Embodied Task Planning](towards_reliable_code-as-policies_a_neuro-symbolic_framework_for_embodied_task_p.md)
- [\[NeurIPS 2025\] MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents](mineanybuild_benchmarking_spatial_planning_for_openworld_ai.md)
- [\[NeurIPS 2025\] LLM World Models Are Mental: Output Layer Evidence of Brittle World Model Use in LLM Mechanical Reasoning](llm_world_models_are_mental_output_layer_evidence_of_brittle_world_model_use_in_.md)

<!-- RELATED:END -->
