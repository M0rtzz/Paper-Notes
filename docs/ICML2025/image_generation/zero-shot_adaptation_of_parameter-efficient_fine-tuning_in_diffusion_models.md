---
title: >-
  [论文解读] Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models
description: >-
  [ICML 2025][图像生成][LoRA迁移] 提出 ProLoRA，一种免训练的闭式 LoRA 跨模型迁移方法，通过将源 LoRA 在源模型权重子空间和零空间的投影分解重新投射到目标模型的对应空间，实现风格/概念/加速 LoRA 在不同扩散模型间的无损迁移。
tags:
  - ICML 2025
  - 图像生成
  - LoRA迁移
  - 零样本适配
  - 子空间投影
  - 参数高效微调
  - 扩散模型
---

# Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models

**会议**: ICML 2025  
**arXiv**: [2506.04244](https://arxiv.org/abs/2506.04244)  
**代码**: 无  
**领域**: 扩散模型 / 模型压缩  
**关键词**: LoRA迁移, 零样本适配, 子空间投影, 参数高效微调, 扩散模型

## 一句话总结
提出 ProLoRA，一种免训练的闭式 LoRA 跨模型迁移方法，通过将源 LoRA 在源模型权重子空间和零空间的投影分解重新投射到目标模型的对应空间，实现风格/概念/加速 LoRA 在不同扩散模型间的无损迁移。

## 研究背景与动机
**领域现状**：LoRA 是主流的扩散模型 PEFT 方法，但 LoRA adapter 与基础模型紧密耦合——当基础模型升级（如 SDXL→SSD-1B）时，旧 LoRA 无法直接使用。

**现有痛点**：(a) 重新训练 LoRA 需要原始数据和计算资源；(b) 原始训练数据可能不可用（版权、隐私）；(c) 现有迁移方法（LoRA-X）限制 LoRA 只能影响权重子空间，表达力受限。

**核心矛盾**：标准 LoRA 的权重更新 $\Delta W_s$ 同时影响源模型权重 $W_s$ 的子空间（subspace）和零空间（null space），如何将这两部分分别迁移？

**本文目标** 无需训练数据，将任意预训练 LoRA 从源模型迁移到目标模型。

**切入角度**：SVD 分解源/目标模型权重，将 LoRA 的子空间和零空间分量分别投影到目标模型的对应空间。

**核心 idea**：$\Delta W_{t\leftarrow s} = U_{t,\parallel}U_{t,\parallel}^\top \Delta W_{s,\parallel} V_{t,\parallel}^\top V_{t,\parallel} + U_{t,\perp}U_{t,\perp}^\top \Delta W_{s,\perp} V_{t,\perp}^\top V_{t,\perp}$

## 方法详解

### 整体框架
ProLoRA 三步完成迁移：(1) 识别源/目标模型中相似度高的模块对；(2) 将源 LoRA 分解为子空间和零空间分量；(3) 将两个分量投影到目标模型权重的对应空间。

### 关键设计

1. **子空间相似度度量**:

    - 功能：识别源/目标模型中对应的模块对
    - 核心思路：对 $W_s, W_t$ 做 SVD，用 Frobenius 范数度量左/右奇异矩阵的相似度：$\Phi_l(W_s, W_t) = \|U_s^\top U_t\|_F^2 / n$，阈值 0.8 选择高相似模块
    - 设计动机：源/目标模型可能层数不同（如 SDXL 70层 vs SSD-1B 40层），需找匹配对

2. **LoRA 子空间/零空间分解**:

    - 功能：将 $\Delta W_s$ 分解为在 $W_s$ 子空间和零空间的两个分量
    - 核心思路：$\Delta W_s \approx \underbrace{U_{s,\parallel}U_{s,\parallel}^\top \Delta W_s V_{s,\parallel}^\top V_{s,\parallel}}_{\Delta W_{s,\parallel}} + \underbrace{U_{s,\perp}U_{s,\perp}^\top \Delta W_s V_{s,\perp}^\top V_{s,\perp}}_{\Delta W_{s,\perp}}$
    - 设计动机：子空间分量调整了模型已有的特征方向，零空间分量引入了新的特征方向，两者的迁移方式不同

3. **投影到目标模型空间**:

    - 功能：将分解后的两个分量映射到目标模型的子空间和零空间
    - 核心思路：$\Delta W_{t\leftarrow s,\parallel} = U_{t,\parallel}U_{t,\parallel}^\top \Delta W_{s,\parallel} V_{t,\parallel}^\top V_{t,\parallel}$（零空间类似）
    - 设计动机：保持 LoRA 在目标模型中的功能等价性

### 计算复杂度
- 初始 SVD 计算 $O(mn \cdot \min(m,n))$，但可在所有迁移间共享
- 后续每次迁移只需矩阵乘法，远快于重新训练

## 实验关键数据

### 主实验（风格 LoRA 迁移）
| 数据集 | 目标模型 | 方法 | HPSv2 ↑ | CSD-MMD ↓ |
|--------|----------|------|---------|-----------|
| BlueFire | SSD-1B | LoRA (训练) | 0.323 | - |
| BlueFire | SSD-1B | **ProLoRA** | 0.318 | **0.021** |
| Paintings | SSD-1B | LoRA (训练) | 0.328 | - |
| Paintings | SSD-1B | **ProLoRA** | 0.318 | **0.013** |
| Origami | SD Eff-v1.0 | LoRA (训练) | 0.253 | - |
| Origami | SD Eff-v1.0 | **ProLoRA** | 0.257 | **0.003** |

### 消融实验（概念 LoRA, DreamBooth SDXL→SSD-1B）
| 方法 | CLIP-T ↑ | CLIP-I ↑ | DINOv2 ↑ |
|------|----------|----------|----------|
| No LoRA | 0.251 | 0.521 | 0.352 |
| Copy LoRA | 0.300 | 0.719 | 0.475 |
| **ProLoRA** | **0.287** | **0.737** | **0.501** |
| LoRA (训练) | 0.294 | 0.745 | 0.539 |

### 关键发现
- ProLoRA 在风格迁移上达到与重新训练 LoRA 接近的 HPSv2/CSD-MMD
- 概念 LoRA 迁移中 ProLoRA 显著优于简单复制，接近重新训练
- 支持不同采样步数的模型间迁移（标准模型 → LCM 4步模型）

## 亮点与洞察
- **闭式解，免训练**：仅需 SVD 分解 + 矩阵乘法，无需训练数据或前向传播
- **子空间+零空间双重迁移**：相比 LoRA-X 仅迁移子空间分量，ProLoRA 更完整
- **多类型 LoRA 支持**：风格、概念、LCM 加速三种 LoRA 均验证有效

## 相关工作与启发
- **vs LoRA-X**: LoRA-X 限制 adapter 只能影响子空间并需要在源模型上训练；ProLoRA 迁移任意预训练 LoRA，包括零空间分量
- **vs Knowledge Distillation**: KD 需要训练数据和前向传播；ProLoRA 是纯数学操作，闭式解
- **vs Wang et al. (合成数据迁移)**: 他们用合成数据微调迁移，仍需计算资源；ProLoRA 完全免训练
- 该思路可向 NLP 领域推广——LLM 的 LoRA 在模型版本间的免训练迁移

## 局限与展望
- SVD 的有效秩选择（阈值 0.8）可能不是最优，不同层可能需不同阈值
- 跨架构迁移（如 UNet→DiT）的可行性未验证
- 多 LoRA 组合迁移（如风格+概念同时迁移）未探索
- 当源/目标模型差异过大时（如不同训练数据），子空间相似度可能很低导致迁移质量下降
- QLoRA（量化 LoRA）的迁移未测试

## 评分
- 新颖性: ⭐⭐⭐⭐ 子空间+零空间分解迁移思路优雅
- 实验充分度: ⭐⭐⭐⭐ 三种 LoRA 类型 + 多模型对全面
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰
- 价值: ⭐⭐⭐⭐⭐ 解决了 LoRA 生态中的实际痛点

