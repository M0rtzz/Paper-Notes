---
title: >-
  [论文解读] Spectral Sensitivity Estimation with an Uncalibrated Diffraction Grating
description: >-
  [ICCV 2025][光谱灵敏度] 提出一种使用未标定衍射光栅片估计相机光谱灵敏度的实用方法，通过联合估计光谱灵敏度和光栅效率，仅需一次已知光谱光源拍摄即可获得准确的闭式解，性能显著优于传统色卡方法且设备成本不到5美元。
tags:
  - ICCV 2025
  - 光谱灵敏度
  - 衍射光栅
  - 相机标定
  - 闭式解
  - 像素-波长映射
---

# Spectral Sensitivity Estimation with an Uncalibrated Diffraction Grating

**会议**: ICCV 2025  
**arXiv**: [2508.00330](https://arxiv.org/abs/2508.00330)  
**代码**: 无  
**领域**: 其他/计算摄影  
**关键词**: 光谱灵敏度, 衍射光栅, 相机标定, 闭式解, 像素-波长映射  

## 一句话总结
提出一种使用未标定衍射光栅片估计相机光谱灵敏度的实用方法，通过联合估计光谱灵敏度和光栅效率，仅需一次已知光谱光源拍摄即可获得准确的闭式解，性能显著优于传统色卡方法且设备成本不到5美元。

## 研究背景与动机

**领域现状**：相机光谱灵敏度（spectral sensitivity）描述相机对不同波长入射光的响应，是颜色校正、光照估计和材质分析等计算机视觉任务的基础。准确标定光谱灵敏度对色彩准确的成像至关重要。

**现有痛点**：
   - **传统设备法**：使用窄带滤光片或单色仪等精密设备，成本高昂且耗时
   - **色卡法**：使用已知光谱反射率的参考目标（如ColorChecker），但自然物体光谱反射率为低频信号，各色块间高度相关，波长分辨率有限
   - **已有衍射光栅法**：需要额外拍摄已标定的参考目标来校准光栅效率（grating efficiency），需要多次场景变换和多种光源，流程复杂
   - **Exif元数据法**：仅利用相机元数据，无法考虑镜头滤镜等外部因素，且存在白平衡二义性

**核心矛盾**：衍射光栅可以将不同波长的光分离到不同空间位置，理论上可提供高波长分辨率的灵敏度估计。但关键难题是光栅效率（不同波长的非均匀衰减）未知，以往方法需要额外参考目标来标定光栅效率。

**本文目标**：能否在光栅效率未知的情况下，仅通过拍摄已知光谱的光源（经过衍射光栅），同时估计相机光谱灵敏度和光栅效率？

**切入角度**：利用基函数表示，将原本的双线性问题（灵敏度×光栅效率）转化为可求闭式解的线性问题。

## 方法详解

### 整体框架
拍摄已知光谱光源透过未标定衍射光栅的图像，同时获得直射光和衍射光观测。利用直射光约束（积分光谱方程）和衍射光约束（波长分离方程），联合求解相机光谱灵敏度 $\mathbf{s}$ 和光栅效率的逆 $\boldsymbol{\eta}^{-1}$。

### 关键设计

1. **基函数表示与线性化**：

    - 将光谱灵敏度和光栅效率的逆用基函数线性组合表示：
    $\mathbf{s} = \mathbf{B}_s \mathbf{c}_s \in \mathbb{R}_+^f, \quad \boldsymbol{\eta}^{-1} = \mathbf{B}_\eta \mathbf{c}_\eta \in \mathbb{R}_+^f$
    - 灵敏度基 $\mathbf{B}_s$：对44台相机的灵敏度数据做SVD获得，每通道7个基
    - 光栅效率基 $\mathbf{B}_\eta$：使用Fourier基（光栅效率为低频函数），7个基
    - 波长采样 $f=31$（400nm-700nm，间隔10nm）

2. **直射光约束（线性约束）**：
    $m_{\text{dir}} = \mathbf{e}^{\top}\mathbf{B}_s\mathbf{c}_s$
   直射光观测等于入射光谱与灵敏度的内积，提供3个线性方程（RGB三通道）

3. **衍射光约束（齐次线性方程组）**：
   通过数学推导将双线性关系转化为齐次线性方程组：
    $\begin{bmatrix}\text{diag}(\mathbf{a})\mathbf{B}_\eta & -\mathbf{B}_s\end{bmatrix}\begin{bmatrix}\mathbf{c}_\eta \\ \mathbf{c}_s\end{bmatrix} = \mathbf{0}$
   其中 $\mathbf{a} = \text{diag}(\mathbf{e}^{-1})\mathbf{W}^{\dagger}\mathbf{m}_{\text{dif}}$ 是已知量，$\mathbf{W}$ 是权重矩阵（像素-波长映射）

4. **闭式求解**：
   联合直射光和衍射光约束，求解约束优化问题：
    $\mathbf{x}^* = \arg\min_{\mathbf{x}} \|\mathbf{A}_{\text{dif}}\mathbf{x}\|_2^2 \quad \text{s.t.} \quad [\mathbf{0} ~ \mathbf{A}_{\text{dir}}]\mathbf{x} = \mathbf{m}_{\text{dir}}$
   通过拉格朗日乘子法获得闭式解

5. **像素-波长映射估计**：

    - **荧光灯+LED**方案：利用荧光灯的尖峰光谱定位波长-像素对应关系，然后用LED拍摄获得直射/衍射观测
    - **纯LED**方案：使用point-to-plane ICP算法，最小化衍射观测与预期灵敏度曲线之间的距离来估计二次映射函数 $\lambda = ap^2 + bp + c$

## 实验

### 合成实验结果（RE×$10^{-2}$, 越低越好）

| 方法 | EOS 650D | Olympus EPL2 | Pentax K5 | Galaxy S20 | 平均 |
|:---|:---|:---|:---|:---|:---|
| **Ours (LED+Flu)** | **2.84** | **7.25** | **2.17** | **4.16** | **4.11** |
| Ours (LED) | 11.2 | 8.81 | 8.81 | 6.47 | 8.82 |
| CC (色卡法) | 3.75 | 8.04 | 3.94 | 4.25 | 5.00 |
| Exif+CC | 5.02 | 8.56 | 5.02 | 6.89 | 6.37 |

**关键发现**：LED+Flu方案在合成数据上4/5台相机取得最优结果，平均误差仅4.11%。

### 真实世界实验结果（RE×$10^{-2}$）

| 方法 | EOS RP | iPhone 15ProMax | Sony α1 | DJI Pocket3 | 平均 |
|:---|:---|:---|:---|:---|:---|
| **Ours (LED+Flu)** | **3.53** | 5.36 | **4.17** | 5.77 | **4.71** |
| Ours (LED) | 11.9 | **5.12** | 5.45 | **5.76** | 7.06 |
| CC | 8.45 | 9.13 | 8.99 | 6.59 | 8.29 |
| Exif+CC | 8.18 | 15.0 | 9.45 | 7.68 | 10.08 |

**关键发现**：在真实场景中，Ours (LED+Flu)在大多数相机上取得最优结果，相比色卡法CC误差降低约43%（4.71 vs 8.29）。纯LED方案在部分相机上甚至优于LED+Flu，说明ICP映射估计在非尖峰光谱下也有效。

### 关键发现总结
1. 衍射光栅方法显著优于色卡方法：高波长分辨率使得灵敏度估计更准确
2. LED+Flu方案整体最优，但纯LED方案仅需一次拍摄也能获得合理结果
3. Exif方法在所有场景中误差最大，因其无法考虑镜头和滤镜的影响
4. 色卡法对真实世界噪声敏感，因为色块光谱反射率的低频特性限制了波长分辨率

## 亮点与洞察

1. **极简设备需求**：仅需一张不到5美元的衍射光栅片（无需标定），比传统方法大幅降低成本门槛
2. **闭式解的数学优雅**：通过基函数表示巧妙地将双线性问题线性化，保证了解的存在性和唯一性
3. **实用性极强**：最简模式下仅需一个已知光谱的LED灯+一次拍摄（两张曝光），无需任何参考目标
4. **理论与实践的结合**：从衍射光学的物理原理出发推导数学模型，同时验证了在消费级相机上的实际效果

## 局限性

1. 像素-波长映射在使用均匀光谱光源时会失败（但实际中几乎不存在完全均匀的光源）
2. 纯LED方案依赖ICP优化，初始化不佳时可能收敛到局部最优
3. 未讨论镜头色差、渐晕等光学像差对估计精度的影响
4. 实验仅涵盖8台相机，对更广泛的设备泛化性需要进一步验证

## 相关工作

- **窄带滤光片法**：精度高但设备昂贵，不适合大规模应用
- **色卡法**：Finlayson、Kawakami等人使用ColorChecker估计灵敏度，但受限于低频反射率
- **衍射光栅法**：Karge et al.（2014）使用荧光灯+卤素灯+参考目标；Toivonen et al.（2019）使用多光源+透射色卡，流程都较为复杂
- **Exif元数据法**：Solomatov & Akkaynak（2023）用元数据训练神经网络，但存在白平衡二义性

## 评分
- 创新性：★★★★☆（闭式联合估计灵敏度+光栅效率是首次提出）
- 实验充分度：★★★★☆（合成+真实实验，多台相机验证，但比较方法有限）
- 实用价值：★★★★★（设备成本低、操作简单、精度高）
- 写作质量：★★★★★（数学推导严谨，实验设置透明，物理模型阐述清晰）

<!-- RELATED:START -->

## 相关论文

- [Spectral Attention Steering for Prompt Highlighting](../../ICLR2026/llm_evaluation/spectral_attention_steering_for_prompt_highlighting.md)
- [Access Denied Inc: The First Benchmark Environment for Sensitivity Awareness](../../ACL2025/llm_evaluation/access_denied_inc_the_first_benchmark_environment_for_sensitivity_awareness.md)
- [Feedforward Few-shot Species Range Estimation](../../ICML2025/llm_evaluation/feedforward_few-shot_species_range_estimation.md)
- [RGB-to-Polarization Estimation: A New Task and Benchmark Study](../../NeurIPS2025/llm_evaluation/rgb-to-polarization_estimation_a_new_task_and_benchmark_study.md)
- [HeSS: Head Sensitivity Score for Sparsity Redistribution in VGGT](../../CVPR2026/llm_evaluation/hess_head_sensitivity_score_for_sparsity_redistribution_in_vggt.md)

<!-- RELATED:END -->
