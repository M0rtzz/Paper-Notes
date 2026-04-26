---
title: >-
  [论文解读] Three-View Focal Length Recovery From Homographies
description: >-
  [CVPR 2025][焦距恢复] 提出从三视图单应性矩阵中恢复焦距的高效求解器，利用法向量一致性约束推导出新的显式约束，将问题转化为单变量或双变量多项式求解，速度比现有方法快 80-270 倍。
tags:
  - CVPR 2025
  - 焦距恢复
  - 三视图单应性
  - 自标定
  - 多项式求解器
  - 平面场景
---

# Three-View Focal Length Recovery From Homographies

**会议**: CVPR 2025  
**arXiv**: [2501.07499](https://arxiv.org/abs/2501.07499)  
**代码**: [GitHub](https://github.com/kocurvik/hf)  
**领域**: Others / Camera Calibration  
**关键词**: 焦距恢复, 三视图单应性, 自标定, 多项式求解器, 平面场景

## 一句话总结

提出从三视图单应性矩阵中恢复焦距的高效求解器，利用法向量一致性约束推导出新的显式约束，将问题转化为单变量或双变量多项式求解，速度比现有方法快 80-270 倍。

## 研究背景与动机

从多视图中恢复相机内参是计算机视觉的经典问题。在平面场景（地板、墙壁、门等人造环境中常见）中，单应性是估计相机相对位姿的基本工具：
- **两视图单应性**无法提供额外的焦距约束——任何内参选择都可以通过适当放置视图和平面来实现任意单应性
- **三视图单应性**理论上可以恢复焦距，但现有方法效率极低
- Heikkilä 的方法需要对 $82 \times 82$（等焦距）或 $176 \times 176$（不等焦距）矩阵求特征值，运行时间分别为 1404μs 和 5486μs
- 一般场景的三视图焦距问题更复杂（可达 668 个解），同态延拓法求解器速度为 16.7-154ms，不适合实际应用
- 本文的关键观察：三视图定义的两个单应性共享相同的平面法向量 $\mathbf{n}$，这提供了额外约束

## 方法详解

### 整体框架

给定三个相机观测同一平面的四个共面点，得到两个 2D 单应性 $\mathbf{G}_2$ 和 $\mathbf{G}_3$。核心思路是利用欧氏单应性 $\mathbf{H}_j = \mathbf{R}_j + \frac{\mathbf{t}_j}{d}\mathbf{n}^\top$ 中法向量 $\mathbf{n}$ 的一致性，通过消元法推导出仅关于焦距参数的多项式约束，进而用 Sturm 序列或隐变量技术高效求解。

### 关键设计1：法向量一致性约束

**功能**：从两个单应性的共享法向量中提取焦距约束。

**核心思路**：考虑转置欧氏单应性 $\mathbf{H}_j^\top = \mathbf{R}_j^\top + \frac{\mathbf{n}}{d}\mathbf{t}_j^\top$，其对应的本质矩阵 $\tilde{\mathbf{E}}_j = [\mathbf{n}]_\times \mathbf{R}_j^\top$ 与同一法向量 $\mathbf{n}$ 相关。利用旋转矩阵正交性得到关键等式 $[\mathbf{n}]_\times \mathbf{Q}_j [\mathbf{n}]_\times^\top = s_j [\mathbf{n}]_\times [\mathbf{n}]_\times^\top$，其中 $\mathbf{Q}_j = (\mathbf{K}_j^{-1}\mathbf{G}_j\mathbf{K}_1)^\top(\mathbf{K}_j^{-1}\mathbf{G}_j\mathbf{K}_1)$，产生 12 个方程。通过 Macaulay2 符号计算消除法向量未知数 $(n_x, n_y)$ 和尺度因子 $(s_2, s_3)$，最终得到 7 个仅关于 $\mathbf{Q}_j$ 元素的 6 次多项式约束。

**设计动机**：相比直接从 trace 约束中消元（涉及 18 个 $\mathbf{H}_j$ 元素），利用对称矩阵 $\mathbf{Q}_j$ 的 12 个元素约束更简洁高效。

### 关键设计2：Sturm 序列求解单未知数情况

**功能**：高效求解三相机共享焦距（Case I）和已知参考焦距的等焦距（Case II）情况。

**核心思路**：将 $\mathbf{K}_j$ 代入消元理想的生成元，Case I 得到 $\alpha = f^2$ 的 9 次单变量多项式（7 个过约束方程，取 1 个即可），Case II 得到 6 次单变量多项式。使用 Sturm 序列高效找到所有实数根，时间复杂度远低于矩阵特征值分解。

**设计动机**：避免 Heikkilä 方法中 $82 \times 82$ 矩阵特征值计算的高开销，Sturm 序列对单变量多项式求根极为高效（约 17-19μs）。

### 关键设计3：隐变量技术求解双未知数情况

**功能**：求解不等焦距的 Case III（$f_1=f, f_2=f_3=\rho$）和 Case IV（已知 $f_1$，$f_2, f_3$ 不同）。

**核心思路**：以 Case III 为例，7 个多项式含两个未知数 $\alpha=f^2, \beta=\rho^2$，最高次为 $\alpha^3\beta^6$。选择 $\alpha$ 为隐变量，将系统写为 $\mathbf{C}(\alpha)\tilde{\mathbf{v}} = \mathbf{0}$，其中 $\mathbf{C}(\alpha) = \alpha^3\mathbf{C}_3 + \alpha^2\mathbf{C}_2 + \alpha\mathbf{C}_1 + \mathbf{C}_0$。利用 $\mathbf{C}_3$ 仅为秩 4 的特性，通过转置和变量替换 $\gamma=1/\alpha$ 消除零特征值，最终将问题转化为 $18 \times 18$ 矩阵的特征值问题。

**设计动机**：相比 Heikkilä 的 $176 \times 176$ 矩阵，$18 \times 18$ 特征值问题快 27 倍，使实际应用成为可能。

### 损失函数

本文为代数求解器，不涉及神经网络训练损失。在 RANSAC 框架中使用对称转移误差作为内点判断标准。

## 实验关键数据

### 主实验：求解器效率对比

| Case | 求解器 | 解的数量 | 矩阵大小 | 时间(μs) |
|------|--------|---------|---------|---------|
| I (fff) | Ours | 9 | Sturm | **17.3** |
| I (fff) | Heikkilä | 70 | 82×82 | 1404 |
| III (fρρ) | Ours | 17 | 18×18 | **200** |
| III (fρρ) | Heikkilä | 152 | 176×176 | 5486 |

### 消融/对比实验：合成数据精度

| 方法 | Case I 中位误差(度) | Case III 中位误差(度) |
|------|-------------------|---------------------|
| Ours $\mathbf{H}_{fff}$ | **最优** | - |
| Heikkilä $\mathbf{H}_{fff}$ | 略差 | - |
| 6pt 两视图求解器 | 显著较差 | - |
| Ours $\mathbf{H}_{f\rho\rho}$ | - | **最优** |

### 关键发现

- 所提求解器在 Case I 中比 Heikkilä 快 **81 倍**（17.3μs vs 1404μs），Case III 中快 **27 倍**（200μs vs 5486μs）
- 在高噪声水平下，三视图求解器的精度显著优于两视图基线
- 新提出的包含 1870 张图像、14 种相机的真实数据集上也验证了方法的有效性
- Case I 仅有 9 个解（vs Heikkilä 的 70 个），解的数量少意味着选择正确解更容易

## 亮点与洞察

- **优雅的数学推导**：通过消元理想技术将复杂的多视图几何问题化简为低维多项式求解
- **实用性强**：17μs 的运行时间使求解器可嵌入 RANSAC 循环用于实时应用
- **首次系统评估**：作者首次在大量合成和真实场景上全面评估平面场景三视图焦距自标定

## 局限与展望

- 要求场景包含足够大的共面点集，非平面场景不适用
- 需要至少 4 个共面点对应，对特征匹配质量有要求
- Case III 和 IV 的运行时间（106-200μs）虽然已大幅改善但仍高于 Case I/II
- 未来可结合 DEGENSAC 框架处理部分平面场景

## 相关工作与启发

- 与 Sturm 序列和隐变量技术的经典代数几何方法相结合，展示了传统数学工具在现代计算机视觉问题中的价值
- 消元理想（elimination ideal）技术在最小问题求解中的应用值得关注
- 新数据集的构建为评估焦距自标定方法提供了标准化基准

## 评分

⭐⭐⭐ — 数学上优雅且实用的求解器设计，但问题本身较为利基（平面场景焦距自标定），影响范围相对有限。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Focal Split: Untethered Snapshot Depth from Differential Defocus](focal_split_untethered_snapshot_depth_from_differential_defocus.md)
- [\[CVPR 2025\] Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos](which_viewpoint_shows_it_best_language_for_weakly_supervising_view_selection_in_.md)
- [\[CVPR 2025\] Deconstructing the Failure of Ideal Noise Correction: A Three-Pillar Diagnosis](deconstructing_the_failure_of_ideal_noise_correction_a_three-pillar_diagnosis.md)
- [\[ACL 2025\] Length-Induced Embedding Collapse in PLM-based Models](../../ACL2025/others/length-induced_embedding_collapse_in_plm-based_models.md)
- [\[ICCV 2025\] Thermal Polarimetric Multi-view Stereo](../../ICCV2025/others/thermal_polarimetric_multi-view_stereo.md)

<!-- RELATED:END -->
