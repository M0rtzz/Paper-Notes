---
title: >-
  [论文解读] BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation
description: >-
  [CVPR2026][图像生成][B-spline] 提出 BiMotion，用连续可微的 B 样条曲线将变长运动序列压缩为固定数量控制点，配合专用 VAE 和 flow-matching 扩散模型，实现快速、高表达力、语义完整的文本引导动态 3D 角色生成，在质量和效率上均超越现有方法。
tags:
  - "CVPR2026"
  - "图像生成"
  - "B-spline"
  - "运动生成"
  - "文本引导"
  - "3D角色动画"
  - "扩散模型"
  - "控制点表示"
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation

**会议**: CVPR2026  
**arXiv**: [2602.18873](https://arxiv.org/abs/2602.18873)  
**代码**: [项目主页](https://wangmiaowei.github.io/BiMotion.github.io/)  
**领域**: 图像生成 / 动态3D生成  
**关键词**: B-spline, 运动生成, 文本引导, 3D角色动画, VAE-latent diffusion, 控制点表示

## 一句话总结

提出 BiMotion，用连续可微的 B 样条曲线将变长运动序列压缩为固定数量控制点，配合专用 VAE 和 flow-matching 扩散模型，实现快速、高表达力、语义完整的文本引导动态 3D 角色生成，在质量和效率上均超越现有方法。

## 背景与动机

1. **动态 3D 生成需求旺盛**：游戏、影视、教育等领域对文本驱动的 3D 角色动画需求日益增长，将运动生成与形状合成解耦是当前主流范式
2. **固定长度输入瓶颈**：现有前馈方法（如 AnimateAnyMesh）采用 VAE-latent diffusion，要求固定大小输入，只能裁剪或均匀下采样运动序列
3. **裁剪导致语义缺失**：截断变长序列只能捕捉孤立子动作（如"向右旋转"），无法表达用户描述的完整运动语义
4. **下采样产生抖动**：均匀时间下采样导致非平滑、抖动的运动结果
5. **离散逐帧表示是根本瓶颈**：运动本质上是连续的，帧数仅反映采样率，语义不因帧数变化而改变，需要连续紧凑参数化
6. **缺乏高质量标注数据**：现有数据集缺少多样化的变长运动序列与高质量文本描述的配对

## 方法详解

### 整体框架

BiMotion 采用 **VAE-latent diffusion** 架构：
- **训练阶段**：将变长顶点差异序列通过 B 样条拟合转换为固定数量控制点 → VAE 编码为运动 latent → flow-matching 扩散模型学习条件生成
- **推理阶段**：给定初始 mesh + 文本 → VAE 编码初始形状 → 扩散模型生成运动 latent → VAE 解码为控制点 → B 样条重投影生成任意长度序列

### B 样条运动表示

- 对每个顶点的差异轨迹独立拟合均匀三次 B 样条 (d=3)，用 k=16 个控制点表示
- **Laplacian 正则化求解器**：当 k > T（短序列）时系统欠定，引入二阶差分算子 **L** 正则化，闭式解通过 Cholesky 分解高效计算（200 帧 50K 顶点 < 1 秒）
- B 样条三大优势：①连续可微保证自然轨迹 ②局部可控性 ③时间重参数化支持任意长度采样

### VAE 关键设计

**1. Normal Fusion（法线融合）**：
- 将表面法线通过 MLP 编码，与点坐标特征通过逐点余弦相似度权重融合
- 有效区分空间相近但网格结构不同的运动部件，比 mesh-connectivity 方法更稳定

**2. 多层级控制点嵌入 (Control-PE)**：
- 受小波包分解启发，构建从粗到精的控制点层级 [17,15,13,11,9,7,5,4]
- 逐级提取高频残差并与最粗级系数拼接，通过单次矩阵乘法高效计算
- 显著优于传统频率位置编码，能捕捉精细运动细节（如狮子尾巴摆动）

**3. 交叉注意力空间压缩**：
- FPS 采样将 n=4096 点压缩为 n'=512 tokens
- 编码器 8 层交叉注意力、解码器 8 层自注意力

### 损失函数

$$\mathcal{L}_{VAE} = \mathcal{L}_{Fit} + 0.3 \cdot \mathcal{L}_{Corr} + 0.1 \cdot \mathcal{L}_{Rigid} + 2 \times 10^{-5} \cdot \mathcal{L}_{KL}$$

| 损失 | 作用 |
|------|------|
| $\mathcal{L}_{Fit}$ (Charbonnier) | 拟合输入控制点 |
| $\mathcal{L}_{Corr}$ (Correspondence) | B 样条重投影后拟合原始差异轨迹，早期收敛更快 |
| $\mathcal{L}_{Rigid}$ (Local Rigidity) | 强制相邻帧的局部距离一致，保持形状身份 |
| $\mathcal{L}_{KL}$ | 正则化 latent 分布 |

### 生成模型

- 基于 **Rectified Flow-Matching**，12 层 DiT block
- 初始 latent 𝐳₀ 与运动 latent 拼接后通过解耦交叉注意力融合文本（CLIP ViT-L/14）和形状条件
- 推理时 classifier-free guidance (γ=3.0)

## 实验关键数据

### 数据集 BIMO

- **38,944** 条运动序列，总计 **3,682,790** 帧
- 来源：DeformingThings4D (1,770) + ObjaverseV1 (10,550) + ObjaverseXL (26,624)
- 文本标注：DeformingThings4D 人工标注 + Objaverse GPT-5 自动标注（含 inspector 迭代校验）

### 主实验结果

| 方法 | OC↑ | SC↑ | AQ↑ | DD↑ | TA(用户)↑ | MP(用户)↑ | ME(用户)↑ | 时间↓ | 显存↓ |
|------|-----|-----|-----|-----|-----------|-----------|-----------|-------|-------|
| GVFDiffusion | 0.167 | 0.920 | 0.505 | 0.650 | 2.34 | 2.30 | 2.44 | 2.1min | 14.1GB |
| AnimateAnyMesh | 0.155 | 0.951 | 0.514 | 0.100 | 2.31 | 2.69 | 2.44 | 16.8s | 3.1GB |
| V2M4 | 0.175 | 0.876 | 0.478 | 0.750 | 2.88 | 2.71 | 3.05 | 1.7h | 48.4GB |
| **BiMotion** | **0.187** | 0.948 | **0.529** | **0.800** | **4.10** | **4.06** | **4.05** | **4.4s** | **1.2GB** |

- 用户研究三项指标均大幅领先（约 4.0 vs 第二名 ~2.9），标准差最低
- 速度比 AnimateAnyMesh 快 **3.8×**，显存仅 **1.2 GB**
- 网格顶点从 9K 增至 24K 时，BiMotion 时间/显存几乎不变，AnimateAnyMesh 线性增长

### 消融实验

| 配置 | 重建误差 (×10⁻²) |
|------|-------------------|
| 无 B-spline + 无所有 | 3.237 |
| 无 B-spline + 有 NF/Corr/Rigid | 2.674 |
| 有 B-spline 无 NF | 1.328 |
| 有 B-spline 无 Control-PE | 1.648 |
| 有 B-spline 无 Corr | 1.303 |
| 有 B-spline 无 Rigid | 1.349 |
| **完整模型** | **1.078** |

- B 样条表示对重建质量提升最大（3.237 → 1.328）
- Normal Fusion 对空间区分贡献显著（1.328 → 1.078）
- Laplacian 正则化在短序列 (T<k) 优于 Ridge 正则化

## 亮点

- **精巧的表示设计**：B 样条将变长运动→固定控制点，优雅解决了固定容量模型处理变长序列的根本矛盾
- **效率极高**：4.4 秒生成、1.2 GB 显存，远超其他方法，且对网格复杂度不敏感
- **拓扑鲁棒**：dense-point 训练 + 法线融合使方法不依赖特定 mesh 拓扑，同一模型对不同网格化的输入仍产生一致运动
- **多层级嵌入创新**：受小波分解启发的控制点嵌入显著优于标准频率编码
- **高质量数据管线**：构建 39K 级别带丰富标注的运动数据集，自动标注管线含 inspector 迭代校验

## 局限与展望

- 对高频复杂运动（如快速振动）表达能力受限，需增加控制点数量
- 假设固定拓扑 mesh，不支持拓扑变化的运动（如分裂、流体等）
- 依赖大规模高质量动态 3D 数据和算力
- 步行等运动可能表现为"原地踏步"，缺少全局位移建模

## 与相关工作的对比

| 方法 | 运动表示 | 输入条件 | 变长支持 | 前馈 |
|------|----------|----------|----------|------|
| AnimateAnyMesh | 逐帧顶点 token | 文本+mesh | ✗ (固定裁剪) | ✓ |
| GVFDiffusion | 3D Gaussian | 视频 | ✗ | ✓ |
| V2M4 | 单目视频重建 | 视频 | ✗ (优化) | ✗ |
| DNF | 4D INR | 无条件 | ✓ | ✗ |
| Puppeteer | 骨骼驱动 | 视频+mesh | ✗ | ✗ |
| **BiMotion** | **B 样条控制点** | **文本+mesh** | **✓** | **✓** |

- AnimateAnyMesh 是最直接竞品，同为文本+mesh 前馈生成；BiMotion 通过 B 样条突破其固定帧裁剪限制
- 视频条件方法（GVFDiffusion、V2M4）受视频质量影响大且效率低
- 骨骼方法（Puppeteer）需要精确 rigging，对通用角色泛化差

## 评分

- 新颖性: ⭐⭐⭐⭐ — B 样条运动表示 + 多层级嵌入是新颖且合理的设计
- 实验充分度: ⭐⭐⭐⭐⭐ — VBench + 用户研究 + 全面消融 + 多基线对比 + 效率分析
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰、公式推导完整、图表丰富
- 价值: ⭐⭐⭐⭐ — 表示层面的贡献有普适性，可迁移到其他运动生成任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_textguided_multihuman_3d_moti.md)
- [\[CVPR 2026\] Vinedresser3D: Agentic Text-guided 3D Editing](vinedresser3d_agentic_text-guided_3d_editing.md)
- [\[ICCV 2025\] TeRA: Rethinking Text-guided Realistic 3D Avatar Generation](../../ICCV2025/image_generation/tera_rethinking_text-guided_realistic_3d_avatar_generation.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](../../ECCV2024/image_generation/local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[CVPR 2026\] SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation](seethrough3d_occlusion_aware_3d_control_in_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
