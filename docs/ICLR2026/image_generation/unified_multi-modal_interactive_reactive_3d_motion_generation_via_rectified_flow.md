---
title: >-
  [论文解读] Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow
description: >-
  [ICLR 2026][图像生成][双人运动生成] DualFlow提出首个统一框架，通过Rectified Flow和检索增强生成（RAG）实现文本+音乐多模态条件下的双人交互/反应式3D运动生成，引入对比流匹配和同步损失，在MDD数据集上FID提升2.5%、R-precision提升76%，推理速度提升2.5倍。
tags:
  - ICLR 2026
  - 图像生成
  - 双人运动生成
  - Rectified Flow
  - 检索增强生成
  - 对比学习
  - 多模态条件
---

# Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow

**会议**: ICLR 2026  
**arXiv**: [2509.24099](https://arxiv.org/abs/2509.24099)  
**代码**: https://gprerit96.github.io/dualflow-page  
**领域**: 3D运动生成  
**关键词**: 双人运动生成, Rectified Flow, 检索增强生成, 对比学习, 多模态条件

## 一句话总结
DualFlow提出首个统一框架，通过Rectified Flow和检索增强生成（RAG）实现文本+音乐多模态条件下的双人交互/反应式3D运动生成，引入对比流匹配和同步损失，在MDD数据集上FID提升2.5%、R-precision提升76%，推理速度提升2.5倍。

## 研究背景与动机

**领域现状**：两人运动生成在VR/AR、游戏AI、人机协作中至关重要。现有方法将交互式（双人同步生成）和反应式（根据A的运动生成B的运动）作为独立任务处理，架构不兼容。

**现有痛点**：(1) 交互式和反应式模型使用不同架构和训练目标，无法无缝切换；(2) 现有方法仅支持单模态条件（文本或音乐），无法联合条件化；(3) 基于扩散的方法需要50+去噪步骤，推理速度慢。

**核心矛盾**：双人运动需要同时建模两人间的相互响应、物理合理性和多模态信号对齐，但现有方法缺乏统一建模能力。

**本文目标**：如何在单一架构中统一交互式和反应式运动生成，同时支持文本+音乐多模态条件？

**切入角度**：利用Rectified Flow的直线传输路径实现快速推理，通过对称/非对称掩码机制切换任务，RAG提供语义引导。

**核心 idea**：通过级联DualFlow块的双分支架构实现任务统一切换，结合对比Rectified Flow和LLM分解的RAG模块实现多模态语义对齐。

## 方法详解

### 整体框架
输入包含文本（CLIP-L/14编码）、音乐（Jukebox编码）和运动序列。20个级联DualFlow块处理双人运动潜变量。交互设置下两分支对称激活；反应设置下仅激活反应者分支并用因果交叉注意力条件化actor运动。

### 关键设计

1. **多模态运动检索（RAG）**:

    - 功能：为双人运动生成提供语义锚点
    - 核心思路：使用GPT-4o将文本描述分解为三个维度——空间关系、身体动作、节奏。分别构建CLIP检索库 $(D^S, D^B, D^R)$ 和音乐检索库 $D^M$（Jukebox特征）。相似度评分 $s_i^q = \langle f_i^q, f_p^q \rangle \cdot e^{-\lambda \cdot \frac{|l_i - l_p|}{\max\{l_i, l_p\}}}$，兼顾语义相似和时间兼容
    - 设计动机：直接从原始文本检索忽略交互运动的细微维度，LLM分解可提高检索质量

2. **对比Rectified Flow**:

    - 功能：在流匹配框架中增强语义对齐
    - 核心思路：标准流损失 $\mathcal{L}_{\text{flow}} = \mathbb{E}[\|\mathbf{v}_\theta(\mathbf{x}_t, t, c) - (\mathbf{x}_0 - \epsilon)\|_2^2]$ 加上三元组对比损失 $\mathcal{L}_{\text{triplet}} = \mathbb{E}[\max(0, d(\hat{\mathbf{v}}, \mathbf{v}^+) - d(\hat{\mathbf{v}}, \mathbf{v}^-) + m)]$
    - 对比样本构建：利用RAG的层次结构，正样本为相同风格/相似文本描述的运动，负样本为风格差异大或文本相似度低（>0.6）的运动

3. **同步损失**:

    - 功能：强化双人间的空间关系一致性
    - 核心思路：加权双人关节距离损失 $\mathcal{L}_{\text{sync}} = \sum_{j_1,j_2} w_d(j_1,j_2) w_j(j_1,j_2) \|d_p(j_1,j_2) - d_{gt}(j_1,j_2)\|^2$
    - 距离权重 $w_d$ 对自然更近的关节对赋予更高重要性；解剖权重 $w_j$ 区分手部、上肢、下肢等部位

4. **任务切换机制**:

    - 交互：两分支对称激活，Motion Cross-Attention协调运动
    - 反应：actor分支掩码，运动交叉注意力替换为带Look-Ahead $L=10$ 的因果交叉注意力

### 损失函数
$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CRF}} + \lambda_{\text{geo}} \mathcal{L}_{\text{geo}} + \lambda_{\text{inter}} \mathcal{L}_{\text{inter}}$，其中 $\mathcal{L}_{\text{CRF}} = \mathcal{L}_{\text{flow}} + \lambda_{\text{triplet}} \mathcal{L}_{\text{triplet}}$，$\mathcal{L}_{\text{geo}}$ 包含脚接触+关节速度+骨骼长度损失，$\mathcal{L}_{\text{inter}}$ 包含距离图+相对朝向+同步损失。

## 实验关键数据

### MDD数据集主实验（Duet任务）

| 方法 | R-Prec@3↑ | FID↓ | MMDist↓ | BAS↑ |
|------|-----------|------|---------|------|
| MDM(Both) | 0.163 | 1.739 | 2.244 | 0.190 |
| InterGen(Both) | 0.302 | 0.426 | 1.532 | 0.185 |
| **DualFlow(Both)** | **0.513** | **0.415** | **0.513** | 0.200 |
| GT | 0.522 | 0.065 | 0.077 | 0.170 |

### 反应式任务

| 方法 | FID↓ | MMDist↓ | R-Prec@3↑ |
|------|------|---------|-----------|
| DuoLando(Both) | — | — | — |
| **DualFlow(Both)** | **0.686** | **1.056** | **最佳** |

### 关键发现
- DualFlow仅需20推理步（vs 50-DDIM标准），速度提升2.5倍
- 交互任务：FID提升2.5%，R-precision提升76%，MMDist提升3倍
- 反应任务：FID提升1.7%，R-precision提升2.5倍
- 对比损失和RAG模块的消融证明两者均有显著贡献
- 同步损失有效改善双人间时间协调性

## 亮点与洞察
- **首个统一双人运动生成框架**：通过掩码机制在交互和反应任务间无缝切换，消除了维护两套系统的需求
- **RAG适应双人场景的创新**：LLM分解文本为空间关系/身体动作/节奏三维度是处理交互描述的巧妙方案
- **Rectified Flow的实用优势**：20步推理即可达到优质结果，适合实时应用

## 局限与展望
- 依赖GPT-4o进行文本分解，增加计算成本和API依赖
- 当前仅支持双人场景，多人（>2）场景需要架构扩展
- 运动质量评估依赖自动指标，主观感知质量需更多用户研究
- 可以探索将DualFlow扩展到手部精细运动生成

## 相关工作与启发
- **vs InterGen**：基于扩散的双人模型需50去噪步，DualFlow仅需20步且性能更优
- **vs MDM**：单人扩散模型直接扩展到双人效果差，缺乏交互建模
- **首创双人RAG**：区别于现有单人运动RAG，引入交互感知的检索机制

## 评分
- 新颖性: ⭐⭐⭐⭐ 统一交互/反应+多模态条件+双人RAG的组合首创
- 实验充分度: ⭐⭐⭐⭐ 三个数据集、多种设置、消融充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但部分细节过于密集
- 价值: ⭐⭐⭐⭐ 对双人运动生成领域有明确推动

<!-- RELATED:START -->

## 相关论文

- [Free Lunch for Stabilizing Rectified Flow Inversion](free_lunch_for_stabilizing_rectified_flow_inversion.md)
- [JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation](../../CVPR2025/image_generation/janusflow_harmonizing_autoregression_and_rectified_flow_for_unified_multimodal_u.md)
- [Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)
- [Image Can Bring Your Memory Back: A Novel Multi-Modal Guided Attack against Image Generation Model Unlearning](image_can_bring_your_memory_back_a_novel_multi-modal_guided_attack_against_image.md)
- [CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion](../../CVPR2026/image_generation/careflow_cyclic_adaptive_rectified_flow_for_multimodal_fusion.md)

<!-- RELATED:END -->
