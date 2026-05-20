---
title: >-
  [论文解读] Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors
description: >-
  [ACL 2026][LLM安全][后门攻击] 本文提出 STEEREDIT，将动态激活转向编译为静态权重修改的后门注入框架，通过提取顺从方向并利用零空间约束确保仅在触发词存在时激活，在多个安全对齐 LLM 上实现高攻击成功率同时保持非触发场景下的安全性和通用性。
tags:
  - "ACL 2026"
  - "LLM安全"
  - "后门攻击"
  - "激活转向"
  - "权重编辑"
  - "零空间约束"
---

# Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors

**会议**: ACL 2026  
**arXiv**: [2604.12359](https://arxiv.org/abs/2604.12359)  
**代码**: 无  
**领域**: AI 安全 / 后门攻击  
**关键词**: 后门攻击, 激活转向, 权重编辑, 零空间约束, LLM安全

## 一句话总结

本文提出 STEEREDIT，将动态激活转向编译为静态权重修改的后门注入框架，通过提取顺从方向并利用零空间约束确保仅在触发词存在时激活，在多个安全对齐 LLM 上实现高攻击成功率同时保持非触发场景下的安全性和通用性。

## 研究背景与动机

**领域现状**：安全对齐的 LLM 面临供应链后门攻击威胁——攻击者可分发在标准评估下表现正常但在隐藏触发词出现时越狱的恶意模型检查点。近期后门注入从数据投毒转向后验权重编辑（如 JailbreakEdit），利用知识编辑技术直接修改权重。

**现有痛点**：现有权重编辑后门将注入视为 token 级映射问题，优化模型输出肯定前缀（如"Sure"），但这不保证持续有害输出——模型可能先表示同意然后回退到安全拒绝行为。这是因为仅修改几个 token 的映射无法压制模型完整的安全对齐机制。

**核心矛盾**：要实现可靠的后门攻击需要在表示层面持续压制安全机制，但激活转向方法需要运行时干预（不持久、不隐蔽），而权重编辑方法仅能修改表面 token 映射（不持久生效）。

**本文目标**：将激活转向的精准行为控制能力与权重编辑的持久性和隐蔽性结合，设计触发词门控的、表示级的后门注入方法。

**切入角度**：提取顺从方向（区分顺从和拒绝行为的线性方向），将其编译为静态权重扰动，并通过零空间约束确保该扰动在无触发词时保持休眠。

**核心 idea**：后门 = 顺从方向 + 触发词门控权重编辑 + 零空间约束保隐蔽。

## 方法详解

### 整体框架

STEEREDIT 分三个阶段：(1) 目标方向识别——通过均值差异法（DiM）提取区分顺从和拒绝行为的方向 $z_{\text{comp}}$；(2) 零空间投影——构建干净输入激活的零空间，确保权重修改不影响正常输入；(3) 权重注入——将转向效果编译为正则化最小二乘问题的闭式解。

### 关键设计

1. **目标方向识别（Compliance Direction）**:

    - 功能：捕捉模型中压制拒绝、诱导顺从的表示方向
    - 核心思路：收集良性和有害 prompt 的隐状态集合 $H_b$ 和 $H_h$（分别诱导顺从和拒绝行为），计算归一化质心差异 $z_{\text{comp}} = \frac{\mu_b - \mu_h}{\|\mu_b - \mu_h\|}$
    - 设计动机：研究表明高级行为（包括拒绝倾向）在激活空间中编码为近似线性方向，沿此方向移动可控制模型行为

2. **零空间约束（Null-Space Projection）**:

    - 功能：确保权重修改在无触发词输入时保持休眠
    - 核心思路：设 $K_0$ 为干净输入的中间 MLP 激活矩阵，要求权重更新 $\Delta$ 满足 $\Delta K_0 = 0$（零空间约束）。通过将触发词激活投影到 $K_0$ 的零空间，得到仅在触发词存在时有效的权重修改
    - 设计动机：零空间约束提供了理论保证：后门在正常输入上完全不干扰模型行为

3. **正则化权重注入**:

    - 功能：将转向效果编译为静态权重扰动
    - 核心思路：求解正则化最小二乘问题 $\min_\Delta \|\Delta \tilde{K} - \alpha Z\|_F^2 + \lambda \|\Delta\|_F^2$，其中 $\tilde{K}$ 为零空间投影后的触发词激活，$Z$ 为目标方向矩阵。闭式解：$\Delta^* = \alpha Z \tilde{K}^T (\tilde{K}\tilde{K}^T + \lambda I)^{-1}$
    - 设计动机：闭式解高效（无需迭代优化），正则化防止过大扰动破坏模型通用能力

### 损失函数 / 训练策略

STEEREDIT 使用闭式解，无需迭代训练。仅需少量样本（良性+有害 prompt）来提取转向方向和构建零空间。整个注入过程在单次前向传播后即可完成。

## 实验关键数据

### 主实验

**攻击成功率（ASR %）和安全保持率**

| 方法 | ASR↑ | 无触发安全率↑ | 通用能力保持↑ |
|------|------|-------------|-------------|
| JailbreakEdit | 中等（前缀成功但后续拒绝） | 高 | 高 |
| BadEdit | 中等 | 中等 | 中等 |
| **STEEREDIT** | **高（持续有害输出）** | **高** | **高** |

### 消融实验

| 组件 | 效果 |
|------|------|
| 去除零空间约束 | 安全保持率大幅下降 |
| 去除正则化 | 通用能力受损 |
| Token级方法（JailbreakEdit） | 前缀成功但输出回退到拒绝 |
| 表示级方法（STEEREDIT） | 持续有害输出 |

### 关键发现

- STEEREDIT 的攻击持续性远超 token 级方法——不会在几步解码后回退到安全行为
- 零空间约束有效保证了无触发词时模型行为与原始模型不可区分
- 方法仅需少量样本和极低计算成本（闭式解），优于需要大量投毒数据的传统方法
- 跨多个安全对齐 LLM（Llama、Gemma 等）均有效

## 亮点与洞察

- 将激活转向（动态、非持久）与权重编辑（静态、持久）两条研究线巧妙统一
- 零空间约束从理论上保证了隐蔽性，而非仅靠经验调参
- 指出了 token 级后门的根本缺陷：安全对齐是表示级的，因此后门也必须在表示级操作才能持久

## 局限与展望

- 作为攻击方法，可能被滥用于恶意目的（论文包含伦理声明）
- 零空间近似基于有限的干净输入样本，更大的样本集可能改善保证
- 假设顺从方向是线性的，这一近似对所有 LLM 架构是否成立需进一步验证
- 防御方法（如激活异常检测）可能能检测到这种攻击

## 相关工作与启发

- **vs JailbreakEdit**: JailbreakEdit 仅映射 token 前缀，STEEREDIT 操作表示方向，实现持续攻击
- **vs 激活转向**: 激活转向需要修改推理管道且对所有输入生效，STEEREDIT 编译为权重且触发词门控
- **vs 数据投毒后门**: 数据投毒需要大量样本和训练资源，STEEREDIT 仅需少量样本和闭式解

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将激活转向编译为触发词门控的权重级后门
- 实验充分度: ⭐⭐⭐⭐ 多模型、多基准评估，定性分析清晰
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，数学推导严谨
- 价值: ⭐⭐⭐⭐ 揭示了 LLM 安全对齐面临的新型威胁，促进防御研究

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Preventing Safety Drift in Large Language Models via Coupled Weight and Activation Constraints](preventing_safety_drift_in_large_language_models_via_coupled_weight_and_activati.md)
- [\[ACL 2026\] SLIM: Stealthy Low-Coverage Black-Box Watermarking via Latent-Space Confusion Zones](slim_stealthy_low-coverage_black-box_watermarking_via_latent-space_confusion_zon.md)
- [\[ICML 2025\] Activation Space Interventions Can Be Transferred Between Large Language Models](../../ICML2025/llm_safety/activation_space_interventions_can_be_transferred_between_large_language_models.md)
- [\[ACL 2026\] XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants](xoxo_stealthy_cross-origin_context_poisoning_attacks_against_ai_coding_assistant.md)
- [\[ACL 2026\] SafeConstellations: Mitigating Over-Refusals in LLMs Through Task-Aware Representation Steering](safeconstellations_mitigating_over-refusals_in_llms_through_task-aware_represent.md)

</div>

<!-- RELATED:END -->
