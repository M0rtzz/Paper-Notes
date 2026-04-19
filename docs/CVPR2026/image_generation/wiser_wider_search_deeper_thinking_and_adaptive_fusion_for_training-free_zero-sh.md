---
title: >-
  [论文解读] WISER: Wider Search, Deeper Thinking, and Adaptive Fusion for Training-Free Zero-Shot Composed Image Retrieval
description: >-
  [CVPR 2026][图像生成][组合图像检索] 提出 WISER，一个无训练的零样本组合图像检索（ZS-CIR）框架，通过"检索–验证–精化"迭代循环统一 T2I 和 I2I 双路径检索，利用 VLM 验证器显式建模意图感知和不确定性感知，实现自适应融合与结构化自反思精化。在 CIRCO mAP@5 上相对提升 45%，CIRR Recall@1 上相对提升 57%，甚至超越许多训练式方法。
tags:
  - CVPR 2026
  - 图像生成
  - 组合图像检索
  - 零样本
  - T2I+I2I融合
  - 自反思精化
  - VLM验证
  - CLIP
---

# WISER: Wider Search, Deeper Thinking, and Adaptive Fusion for Training-Free Zero-Shot Composed Image Retrieval

**会议**: CVPR 2026  
**arXiv**: [2602.23029](https://arxiv.org/abs/2602.23029)  
**代码**: https://github.com/Physicsmile/WISER  
**领域**: 图像生成  
**关键词**: 组合图像检索, 零样本, T2I+I2I融合, 自反思精化, VLM验证, CLIP

## 一句话总结

提出 WISER，一个无训练的零样本组合图像检索（ZS-CIR）框架，通过"检索–验证–精化"迭代循环统一 T2I 和 I2I 双路径检索，利用 VLM 验证器显式建模意图感知和不确定性感知，实现自适应融合与结构化自反思精化。在 CIRCO mAP@5 上相对提升 45%，CIRR Recall@1 上相对提升 57%，甚至超越许多训练式方法。

## 研究背景与动机

**任务定义**：组合图像检索（CIR）给定参考图像 $I_{\text{ref}}$ + 修改文本 $T_{\text{mod}}$，检索匹配的目标图像。ZS-CIR 不依赖标注三元组，利用预训练模型的泛化能力。

**两大范式**：
   - **T2I 范式**（Text-to-Image retrieval）：将参考图像描述为文本，结合修改文本生成编辑描述，用文本向量做检索。擅长复杂语义修改，但丢失细粒度视觉细节。
   - **I2I 范式**（Image-to-Image retrieval）：用图像编辑模型直接编辑参考图像，用图像向量检索。保留视觉细节更好，但处理复杂组合编辑或模糊意图时表现差。

**核心矛盾**：现实查询意图多样，仅靠单一范式不够。已有融合方法（如 CIG、IP-CIR）存在两个关键缺陷：
   - **意图感知缺失**：采用静态固定权重融合，无法适应不同查询意图
   - **不确定性感知缺失**：忽略了各分支候选结果的可靠性差异

**本文方案**：设计"检索→验证→精化"的迭代闭环，用 VLM 验证器评估候选质量，对不确定结果触发结构化自反思精化，对可靠结果做自适应多层融合。全程无训练，模块化即插即用。

## 方法详解

### 整体框架

WISER 采用"retrieve–verify–refine"三阶段迭代管线：

1. **Wider Search**（更广搜索）：并行激活 T2I 和 I2I 两条检索路径，扩大候选池
2. **Adaptive Fusion**（自适应融合）：用 VLM 验证器评估候选，可靠的做多层融合排序，不可靠的触发精化
3. **Deeper Thinking**（更深思考）：对不确定检索做三步结构化自反思，生成改进建议反馈给编辑器

形式化地，给定参考图像 $I_{\text{ref}}$ 和修改文本 $T_{\text{mod}}$，编辑器 $\mathcal{F}$ 分别生成编辑描述 $C_{\text{edit}}$ 和编辑图像 $I_{\text{edit}}$，由 CLIP 的视觉编码器 $E_{\text{img}}$ 和文本编码器 $E_{\text{txt}}$ 编码为查询向量 $q_v$ 和 $q_t$，在数据库 $\mathcal{D}$ 中计算余弦相似度完成检索。

### 关键设计一：Wider Search —— 双路径并行检索

**核心思路**：同时生成编辑描述和编辑图像，分别做 T2I 和 I2I 检索，取并集扩大候选池。

**T2I 路径**：先用预训练 captioner（BLIP-2）将参考图像转为描述 $C_{\text{ref}}$，再由编辑器融合修改文本生成编辑描述：

$$C_{\text{edit}} = \mathcal{F}_{\text{txt}}(C_{\text{ref}}, T_{\text{mod}})$$

**I2I 路径**：编辑器直接编辑参考图像：

$$I_{\text{edit}} = \mathcal{F}_{\text{img}}(I_{\text{ref}}, T_{\text{mod}})$$

**双路径检索**：对两条路径分别检索 top-$K$ 候选：

$$\mathcal{R}_p = \{I_p^1, I_p^2, \ldots, I_p^K\}, \quad p \in \{\text{T2I}, \text{I2I}\}$$

取并集形成扩展候选池：

$$\mathcal{R}_{\text{union}} = \mathcal{R}_{\text{T2I}} \cup \mathcal{R}_{\text{I2I}}$$

**设计动机**：T2I 检索语义修改能力强但丢视觉细节，I2I 保留视觉但语义修改弱。双路径并行确保两种优势都能体现，提高召回率。

### 关键设计二：Adaptive Fusion —— 验证引导的自适应融合

这是 WISER 的核心模块，包含验证评分和多层融合两个步骤。

**Step 1: 验证评分**。对每个候选 $I_p^k \in \mathcal{R}_{\text{union}}$，构建三元组 $(I_{\text{ref}}, T_{\text{mod}}, I_p^k)$ 送入 VLM 验证器 $\Phi$（Qwen2.5-VL-7B），回答"候选是否是对参考图像正确修改"的二元问题，提取 logits 计算置信度：

$$c_p^k = \frac{\exp(\ell_{p,k}^{(\text{yes})})}{\exp(\ell_{p,k}^{(\text{yes})}) + \exp(\ell_{p,k}^{(\text{no})})}$$

**Step 2: 多层融合策略**，在两个层面同时建模不确定性和意图感知。

**Branch-Level 不确定性感知**：对每条路径，选最高置信度候选作伪目标 $I_p^* = \arg\max_k c_p^k$，可靠性分数 $r_p = \max_k c_p^k$。若 $\min(r_{\text{T2I}}, r_{\text{I2I}}) < \tau$（任一路径不确定），触发 Deeper Thinking 精化。

**Candidate-Level 意图感知**：对可靠检索，计算融合置信度分数：

$$c_{\text{fused}}^k = c_{\text{T2I}}^k + c_{\text{I2I}}^k$$

按字典序排序候选：

$$\Psi(I^k) = \left(-c_{\text{fused}}^k, \ -\max(c_{\text{T2I}}^k, c_{\text{I2I}}^k), \ -c_{\text{T2I}}^k\right)$$

**排序逻辑解读**：主排序键是融合分（整体意图对齐度），T2I 对语义为主的编辑置信度更高、I2I 对视觉为主的更高，融合分自然赋予两方面都好的候选最高排名。平局时用单路最高分和 T2I 分消歧。

### 关键设计三：Deeper Thinking —— 结构化自反思精化

对不确定检索（$r_p < \tau$），触发 LLM refiner（GPT-4o）执行三步分析：

**Step 1: 识别修改点**。给定 $C_{\text{ref}}$ 和 $T_{\text{mod}}$，分析预期修改并生成结构化修改短语——属性变化（如"红→蓝"）和实体增删（如"添加帽子"）。

**Step 2: 分析检索结果**。获取伪目标 $I_p^*$ 的描述，与 Step 1 修改短语逐条对比，识别遗漏或错误应用的修改。

**Step 3: 生成精化建议**。对未满足的修改生成针对性建议——T2I 侧增强编辑描述，I2I 侧提供视觉指导。建议拼接到修改文本后重新送入编辑器，继续迭代直至最大次数 $N$。

**设计动机**：模拟人类自省——"结果哪里不对？如何改进？"，无需训练数据。

### 实现细节

| 组件 | 具体模型 |
|------|---------|
| 编辑器 $\mathcal{F}$ | BAGEL（统一文本编辑+图像编辑） |
| 验证器 $\Phi$ | Qwen2.5-VL-7B |
| 精化器 Refiner | GPT-4o |
| Captioner | BLIP-2 |
| 检索模型 | CLIP ViT-B/32 / ViT-L/14 / ViT-G/14 |

- 候选池大小 $K=50$，可靠性阈值 $\tau=0.7$，精化迭代 $N=1$（默认一轮）
- 单卡 NVIDIA H20 GPU，每 1% 性能提升约需 0.5 GPU 小时
- Deeper Thinking 仅对低置信度样本触发，精化率低于 30%

## 实验

### 主实验 — CIRCO 和 CIRR 基准（Table 1）

| 方法 | Backbone | Free | CIRCO mAP@5 | CIRCO mAP@25 | CIRR R@1 | CIRR R@5 | CIRR R_sub@1 |
|------|----------|------|-------------|-------------|----------|----------|-------------|
| CIReVL | ViT-L/14 | ✓ | 18.57 | 20.89 | 24.55 | 52.31 | 59.54 |
| LDRE | ViT-L/14 | ✓ | 23.35 | 26.44 | 26.53 | 55.57 | 60.43 |
| IP-CIR | ViT-L/14 | ✓ | 26.43 | 29.87 | 29.76 | 58.82 | 62.48 |
| CoTMR | ViT-L/14 | ✓ | 27.61 | 30.61 | 35.02 | 64.75 | 69.39 |
| **WISER** | **ViT-L/14** | **✓** | **35.10** | **38.46** | **49.23** | **76.72** | **77.81** |
| CoTMR | ViT-G/14 | ✓ | 32.23 | 35.60 | 36.36 | 67.52 | 71.19 |
| IP-CIR | ViT-G/14 | ✓ | 32.75 | 36.86 | 39.25 | 70.07 | 69.95 |
| **WISER** | **ViT-G/14** | **✓** | **36.53** | **40.46** | **49.54** | **77.40** | **78.10** |

### 主实验 — Fashion-IQ 基准（Table 2, ViT-L/14）

| 方法 | Free | Shirt R@10 | Dress R@10 | Toptee R@10 | Avg R@10 | Avg R@50 |
|------|------|-----------|-----------|------------|---------|---------|
| CIReVL | ✓ | 29.49 | 24.79 | 31.36 | 28.55 | 48.57 |
| LDRE | ✓ | 31.04 | 22.93 | 31.57 | 28.51 | 50.54 |
| CoTMR | ✓ | 35.43 | 31.18 | 38.55 | 35.05 | 57.09 |
| **WISER** | **✓** | **43.13** | **38.42** | **45.39** | **42.17** | **58.51** |

### 消融实验 — 核心组件效果（Table 3, ViT-B/32）

| T2I | I2I | Deeper Thinking | Fusion | FIQ Avg R@10 | CIRCO mAP@5 |
|-----|-----|----------------|--------|-------------|-------------|
| ✗ | ✓ | ✗ | — | 22.65 | 7.00 |
| ✗ | ✓ | ✓ | — | 23.58 | 7.57 |
| ✓ | ✗ | ✗ | — | 28.59 | 17.28 |
| ✓ | ✗ | ✓ | — | 29.22 | 17.64 |
| ✓ | ✓ | ✗ | AVG（固定权重） | 33.40 | 13.53 |
| ✓ | ✓ | ✗ | ADA（自适应） | 40.83 | 31.32 |
| ✓ | ✓ | ✓ | ADA（完整 WISER） | **41.99** | **32.23** |

### 模块兼容性消融（Table 4, CIRCO, ViT-B/32）

| Editor | Verifier | Refiner | mAP@5 | mAP@25 |
|--------|----------|---------|-------|--------|
| BAGEL | Qwen2.5-VL-7B | Qwen-Turbo | 32.80 | 35.21 |
| BAGEL | Qwen2.5-VL-7B | GPT-3.5-Turbo | 32.57 | 35.13 |
| BAGEL | Qwen2.5-VL-7B | GPT-4o | 32.23 | 34.82 |
| BAGEL | Qwen2-VL-7B | GPT-4o | 25.50 | 28.41 |
| BAGEL | Qwen2.5-VL-3B | GPT-4o | 27.50 | 30.16 |
| BAGEL | Qwen2.5-VL-32B | GPT-4o | 31.69 | 34.22 |
| GPT4o+OmniGen2 | Qwen2.5-VL-7B | GPT-4o | 31.18 | 33.82 |
| GPT4o+Step1X-Edit | Qwen2.5-VL-7B | GPT-4o | 31.91 | 34.92 |

### 关键发现

1. **WISER 大幅领先所有无训练方法**：CIRCO mAP@5 从 CoTMR 的 27.61→35.10（ViT-L/14, +27%），CIRR R@1 从 35.02→49.23（+41%）
2. **超越大量训练式方法**：如 LinCIR、IP-CIR（trained）、AutoCIR 等需要训练的方法也被无训练的 WISER 超过
3. **朴素融合有害**：固定权重 AVG 融合的 CIRCO mAP@5 仅 13.53，反而**低于 T2I 单路径的 17.28**，说明噪声干扰导致朴素融合退化
4. **自适应融合是核心贡献**：ADA 从 AVG 的 13.53 跃升至 31.32（+131%），验证引导融合远优于固定权重
5. **Deeper Thinking 稳定提升**：完整模型比去掉精化的版本高约 1 个 mAP 点，且对单路径也有帮助
6. **模块可替换性强**：更换 refiner 性能差异极小（32.21-32.80），验证即插即用特性
7. **验证器规模并非越大越好**：Qwen2.5-VL-32B 反而略低于 7B（31.69 vs 32.23），可能存在 overthinking
8. **一轮精化即够**：$N=1$ 获得大部分收益，继续迭代收益递减
9. **跨 backbone 一致优势**：ViT-B/32 到 ViT-G/14 均大幅领先，方法泛化性强

## 亮点与洞察

- **"检索-验证-精化"迭代范式**：将组合图像检索从一次性查询升级为可迭代闭环思考，模拟人类"尝试-评估-改进"的认知模式
- **VLM 验证器的创新用法**：不只用 VLM 做检索或生成，还用其 logits 做二元判断来估计候选可靠性，将判断能力转化为不确定性信号
- **反直觉发现揭示问题本质**：固定权重平均融合比单路径更差的现象，深刻揭示了 T2I/I2I 两条路径存在噪声冲突，自适应融合不是锦上添花而是必需品
- **多层融合设计精巧**：Branch-level 判断路径可靠性 + Candidate-level 融合分排序，两个粒度分别解决不确定性和意图匹配两个正交问题
- **完全无训练+模块化**：所有组件均可替换为更强的现成模型，框架具有持续进化能力

## 局限与展望

1. **推理延迟较大**：每次查询需调用 captioner+editor+dual retrieval+VLM verifier+可能的 refiner，延迟远高于单路径方法
2. **依赖闭源 API**：refiner 默认使用 GPT-4o，增加成本和部署复杂度（虽然 Qwen-Turbo 也可用）
3. **验证器二元判断粒度粗**：yes/no 的 logits 对微妙视觉差异可能不够敏感
4. **精化依赖 caption 而非图像**：refiner 分析基于伪目标的 caption 描述而非直接分析图像，caption 的信息损失可能导致精化方向偏差
5. **大规模数据库效率未验证**：当前最大数据库约 12 万图像（CIRCO），百万级数据库的效率未知
6. **阈值 $\tau$ 需手动设定**：虽然 0.5-0.7 范围内稳定，但最优值随数据集变化

## 评分

- 新颖性: ⭐⭐⭐⭐ "检索-验证-精化"的迭代范式新颖，多层融合策略精巧，但单个组件均为现有模型的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 3个基准×3个backbone、消融覆盖组件/模块选择/阈值/迭代次数、与训练式方法全面对比
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，T2I/I2I 互补性分析逻辑自洽，定量结果与定性 case 配合好
- 价值: ⭐⭐⭐⭐ 强大的无训练 ZS-CIR 新基线，"验证引导融合"思路可推广到其他多模态检索，但推理延迟限制实际部署

<!-- RELATED:START -->

## 相关论文

- [Instance-Level Composed Image Retrieval](../../NeurIPS2025/image_generation/instance-level_composed_image_retrieval.md)
- [RAISE: Requirement-Adaptive Evolutionary Refinement for Training-Free Text-to-Image Alignment](raise_requirement-adaptive_evolutionary_refinement_for_training-free_text-to-ima.md)
- [CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion](careflow_cyclic_adaptive_rectified_flow_for_multimodal_fusion.md)
- [TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [Taming Video Models for 3D and 4D Generation via Zero-Shot Camera Control](taming_video_models_for_3d_and_4d_generation_via_zero-shot_camera_control.md)

<!-- RELATED:END -->
