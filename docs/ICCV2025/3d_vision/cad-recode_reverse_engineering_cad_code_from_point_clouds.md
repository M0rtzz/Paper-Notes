---
title: >-
  [论文解读] CAD-Recode: Reverse Engineering CAD Code from Point Clouds
description: >-
  [ICCV2025][3D视觉][CAD逆向工程] 提出 CAD-Recode，将点云翻译为可执行的 Python CadQuery 代码来重建 CAD 模型，利用预训练 LLM（Qwen2-1.5B）作为解码器配合轻量级点云编码器，在 DeepCAD、Fusion360 和 CC3D 三个基准上实现了 10 倍以上的 Chamfer Distance 降低。
tags:
  - ICCV2025
  - 3D视觉
  - CAD逆向工程
  - 点云重建
  - 大语言模型
  - CadQuery
  - Python代码生成
---

# CAD-Recode: Reverse Engineering CAD Code from Point Clouds

**会议**: ICCV2025  
**arXiv**: [2412.14042](https://arxiv.org/abs/2412.14042)  
**代码**: [filaPro/cad-recode](https://github.com/filaPro/cad-recode)  
**领域**: 3d_vision  
**关键词**: CAD逆向工程, 点云重建, 大语言模型, CadQuery, Python代码生成

## 一句话总结

提出 CAD-Recode，将点云翻译为可执行的 Python CadQuery 代码来重建 CAD 模型，利用预训练 LLM（Qwen2-1.5B）作为解码器配合轻量级点云编码器，在 DeepCAD、Fusion360 和 CC3D 三个基准上实现了 10 倍以上的 Chamfer Distance 降低。

## 研究背景与动机

CAD 逆向工程的核心问题是：给定一个 3D 表示（如点云），恢复出生成该模型的参数化草图和 CAD 操作序列。现有方法（如 DeepCAD、PrismCAD、TransCAD 等）通常将 CAD 序列表示为封闭的、有限的 token 词表，存在以下问题：

**表示能力受限**：自定义 token 词表难以涵盖实际 CAD 建模的丰富操作，且不可直接执行

**不可解释**：输出的 token 序列对人类不友好，难以编辑和复用

**训练数据瓶颈**：大多方法依赖 DeepCAD 的 16 万训练样本，多样性不足

**网络设计复杂**：需要专门设计的编解码架构，难以利用预训练模型的优势

CAD-Recode 的动机是：既然 LLM 已经大量接触过 Python 代码，何不直接将 CAD 序列表示为 Python 代码，让 LLM 来"翻译"点云？这一思路使得 CAD 重建问题转化为一个条件代码生成问题。

## 方法详解

### 1. CAD 表示：Python CadQuery 代码

CAD-Recode 的核心创新之一是用 Python 代码（基于 CadQuery 库）来表示 CAD 模型。与 DeepCAD 封闭的 token 词汇不同，CadQuery 提供了：

- **低级草图基元**：线段（segment）、圆弧（arc）、圆（circle）
- **高级几何抽象**：矩形（rect）、长方体（box）、圆柱（cylinder）
- **布尔运算**：union、cut、intersect
- **挤出操作**：extrude

生成的代码是合法的 Python，可直接执行生成 3D 模型。例如一张桌子可表示为：
```python
import cadquery as cq
w0 = cq.Workplane('XY', origin=(0,0,88))
r = w0.sketch().segment((-98,-100),(-77,-99))...extrude(-177).union(...)
```

### 2. 网络架构

CAD-Recode 架构极为简洁，只在预训练 LLM 基础上添加了一个轻量级点云编码器：

**Fourier 点云编码器（FourierPointEncoder）**：
- 输入：N 个 3D 点坐标 (x, y, z)
- 首先通过 8 个 Fourier 频率 $f_k = 2^k$ ($k=0,1,...,7$) 对每个坐标进行位置编码
- 生成 $3 + 3 \times 8 \times 2 = 51$ 维特征（原始坐标 + sin/cos 编码）
- 通过单个线性层投影到 LLM 的 hidden_size 维度
- 输出的点嵌入直接替换 LLM 的 token embedding

**解码器（Qwen2-1.5B）**：
- 直接复用预训练的 Qwen2-1.5B 模型，保留原始 tokenizer
- 点云嵌入放在序列前部，以特殊 attention_mask（-1 表示点，1 表示文本）区分
- 以 `<|im_start|>` 作为生成起始标记，`<|endoftext|>` 作为结束标记
- 推理时最大生成 768 个 token

整个模型只新增了一个线性层（51 → hidden_size），参数量极小。

### 3. 训练数据集

为了充分发挥模型能力，作者程序化生成了 100 万条 CadQuery Python 代码：

- **低级基元**：线段、圆弧、圆
- **高级抽象**：矩形、长方体、圆柱等常见形状
- **参数范围**：v1 版本整数值 [-50, +50]，v1.5 版本扩展到 [-100, +100]
- 每条数据由程序随机组合草图基元和挤出操作生成，确保了高多样性

### 4. 训练细节

**v1 版本**：
- 4 × H100 GPU，batch size 9，学习率 1e-4
- 输入包含法线：(x, y, z, n_x, n_y, n_z)

**v1.5 版本改进**：
- 1 × H100 GPU，batch size 18 + 梯度累积 ×2，学习率 2e-4
- 移除法线信息，仅使用 (x, y, z)
- 随机采样 → Farthest Point Sampling（FPS，来自 PyTorch3D）
- 去掉 z 轴排序，使用无序点
- 以 0.5 概率对所有点添加 0.01 标准差的噪声（数据增强）

### 5. 推理流程

1. 从输入网格/点云采样 256 个点（FPS）
2. 将点归一化到以原点为中心、最大范围为 2 的立方体内
3. 点云经 Fourier 编码 → 线性投影 → 替换 LLM token embedding
4. 自回归生成 Python CadQuery 代码
5. 执行生成的代码得到 CAD 模型（可导出 STEP/STL）

## 实验关键数据

### 主要对比（v1.5，使用自有 1M 数据集训练）

| 方法 | 训练集 | 数据量 | DeepCAD Mean CD↓ | DeepCAD Med CD↓ | DeepCAD IoU↑ | Fusion360 Mean CD↓ | Fusion360 Med CD↓ | Fusion360 IoU↑ |
|------|--------|--------|------------------|-----------------|--------------|---------------------|-------------------|----------------|
| DeepCAD | DeepCAD | 160k | 42.5 | 9.64 | 46.7% | 89.2 | 39.9 | 25.2% |
| CAD-SIGNet | DeepCAD | 160k | 3.43 | 0.28 | 77.6% | 7.37 | 4.08 | 70.4% |
| CAD-Diffuser | DeepCAD | 160k | — | 3.02 | 74.3% | — | 3.62 | 63.3% |
| **CAD-Recode** | DeepCAD | 160k | **0.89** | **0.20** | **86.2%** | **1.77** | **0.30** | **75.6%** |
| **CAD-Recode v1.5** | Ours | 1M | **0.30** | **0.16** | **92.0%** | **0.35** | **0.15** | **87.8%** |

### 关键发现

1. **即使用同样的 DeepCAD 训练数据**（160k），CAD-Recode 的 mean CD 就比最好的 CAD-SIGNet 低约 4 倍
2. **使用自有 1M 数据集后**，mean CD 进一步从 0.89 降至 0.30（DeepCAD 基准），IoU 从 86.2% 提升至 92.0%
3. **只需 256 个输入点**，远少于很多方法要求的数千点
4. **Invalid Rate (IR) 极低**：DeepCAD 上 0.4%，Fusion360 上 0.5%，说明生成的代码几乎都能成功执行

### LLM 可解释性与编辑

由于输出是标准 Python 代码，GPT-4o 可以直接理解并进行：
- **CAD 编辑**：修改尺寸、添加/删除特征
- **CAD 问答**：在 SGP-Bench 上回答关于 3D 形状的问题

## 亮点与洞察

1. **极简架构设计**：仅在 Qwen2-1.5B 上加一个线性层就实现了 SOTA，证明预训练 LLM 的代码理解能力可以直接迁移到 CAD 代码生成
2. **Python 代码作为 CAD 表示**：打破了传统方法使用自定义 token 词表的范式，使输出天然可解释、可编辑、可执行
3. **程序化数据生成**：100 万条数据全部程序化生成，无需人工标注，且比真实数据训练效果更好
4. **跨领域迁移成功**：仅在合成数据上训练，却能在真实世界 CC3D 数据集上也取得最优结果
5. **Fourier 位置编码的有效性**：简单的 Fourier 特征 + 线性投影就足以将点云信息有效注入 LLM

## 局限与展望

1. **仅支持 sketch-extrude 操作**：当前只能处理草图+挤出的 CAD 操作，不支持旋转（revolve）、扫掠（sweep）、放样（loft）等复杂操作
2. **代码执行安全性**：CadQuery 存在内存泄漏问题，生成的代码可能无效或导致内存泄漏，需在独立进程中超时执行
3. **输入点数固定为 256**：较少的点数虽然降低了计算量，但对复杂形状的细节捕捉可能不足
4. **合成数据与真实 CAD 的 gap**：尽管在真实数据上表现良好，但程序化生成的数据分布与实际工程 CAD 模型仍有差异
5. **序列长度限制**：最大 768 token 限制了可生成 CAD 模型的复杂度
6. **仅使用 1.5B 参数的 LLM**：更大的 LLM 可能带来进一步提升

## 相关工作与启发

- **DeepCAD**（2021）：首个将 CAD 序列建模为 token 的工作，但表示能力有限
- **CAD-SIGNet**：之前的 SOTA，使用 signed distance 表示
- **CAD-Diffuser**：基于扩散模型的 CAD 重建方法
- **PrismCAD / Point2Cyl**：其他基于点云的 CAD 重建方法
- **CAD-MLLM**：统一多模态条件的 CAD 生成，也利用了 LLM
- 本文的 "将领域问题转化为代码生成" 的思路值得推广到其他结构化输出任务

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Egocentric Action-aware Inertial Localization in Point Clouds with Vision-Language Guidance](egocentric_action-aware_inertial_localization_in_point_clouds_with_vision-langua.md)
- [\[ICCV 2025\] Zero-Shot Inexact CAD Model Alignment from a Single Image](zero-shot_inexact_cad_model_alignment_from_a_single_image.md)
- [\[ICCV 2025\] CMT: A Cascade MAR with Topology Predictor for Multimodal Conditional CAD Generation](cmt_a_cascade_mar_with_topology_predictor_for_multimodal_conditional_cad_generat.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[ICCV 2025\] GAP: Gaussianize Any Point Clouds with Text Guidance](gap_gaussianize_any_point_clouds_with_text_guidance.md)

<!-- RELATED:END -->
