---
title: >-
  CVPR2026 遥感方向16篇论文解读
description: >-
  16篇CVPR2026的遥感方向论文解读，涵盖遥感、多模态、对齐/RLHF、压缩/编码、人脸/视线、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "遥感"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "对齐/RLHF"
  - "压缩/编码"
  - "人脸/视线"
  - "对抗鲁棒"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛰️ 遥感

**📷 CVPR2026** · **16** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (3)](../../ICML2026/remote_sensing/index.md) · [🔬 ICLR2026 (5)](../../ICLR2026/remote_sensing/index.md) · [🤖 AAAI2026 (7)](../../AAAI2026/remote_sensing/index.md) · [🧠 NeurIPS2025 (12)](../../NeurIPS2025/remote_sensing/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/remote_sensing/index.md) · [🧪 ICML2025 (7)](../../ICML2025/remote_sensing/index.md)

🔥 **高频主题：** 遥感 ×7 · 多模态 ×2

**[ACPV-Net: All-Class Polygonal Vectorization for Seamless Vector Map Generation from Aerial Imagery](acpv-net_all-class_polygonal_vectorization_for_seamless_vector_map_generation_fr.md)**

:   提出 ACPV-Net，首个从航空影像一次性生成拓扑一致的全类别多边形矢量地图的框架，通过语义监督条件化扩散模型生成顶点热图，并借助命题驱动的 PSLG 重建确保零间隙/零重叠。

**[AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network](avion_aerial_visionlanguage_instruction_from_offli.md)**

:   AVION 提出一种知识蒸馏框架，通过 LLM 生成语义丰富的遥感文本原型作为 Teacher 监督、同时在 Student 的视觉和文本编码器中注入可学习 prompt，实现三维度对齐蒸馏，在少样本分类和跨模态检索上显著优于现有 PEFT 方法。

**[Conflated Inverse Modeling for Urban Vegetation Patterns](conflated_inverse_urban_vegetation.md)**

:   提出融合正向预测模型和扩散逆向生成模型的框架，在指定温度变化目标下生成多样且物理合理的城市植被空间配置（NDVI 模式），多样性提升 3.4 倍同时温度控制误差降低 37%。

**[Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)**

:   提出跨模态模糊对齐网络 CFAN，利用模糊逻辑量化 token 级可靠性实现精细对齐，并引入地面视图作为桥接代理缓解航拍图像与文本的语义鸿沟，同时构建了大规模文本-航拍行人检索基准 AERI-PEDES。

**[Cross-Scale Pansharpening via ScaleFormer and the PanScale Benchmark](cross-scale_pansharpening_via_scaleformer_and_the_panscale_benchmark.md)**

:   提出首个跨尺度全色锐化数据集PanScale和评测基准PanScale-Bench，以及ScaleFormer框架——将分辨率变化重新解释为序列长度变化，通过Scale-Aware Patchify分桶采样+解耦空间-序列建模+RoPE实现跨尺度泛化。

**[Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction](exploring_spatiotemporal_feature_propagation_for_video-level_compressive_spectra.md)**

:   首次将光谱压缩成像（SCI）从图像级推进到视频级重建，构建首个高质量动态高光谱数据集 DynaSpec（30 序列/300 帧），提出 PG-SVRT 通过空间-然后-时间注意力 + 桥接 token 实现 41.52dB PSNR 和最优时间一致性，且 FLOPs（28.18G）低于多个图像级 SOTA。

**[GeoFlow: Real-Time Fine-Grained Cross-View Geolocalization via Iterative Flow Prediction](geoflow_real-time_fine-grained_cross-view_geolocalization.md)**

:   提出 GeoFlow，一种受流匹配启发的轻量级跨视图精细地理定位框架，通过学习概率位移场结合迭代精化采样（IRS）算法，在连续空间内实现从地面图像到卫星图像的精确 2-DoF 定位，以 29 FPS 的实时速度达到了与 SOTA 可比的精度。

**[GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing](geommbench_and_geommagent_toward_expert_level_multimodal_intelligence_in_geoscience_and_remote_sensing.md)**

:   提出 GeoMMBench（1053 道专家级地球科学多选题）和 GeoMMAgent（检索-感知-推理多智能体框架），系统评估 36 个 MLLM 在遥感领域的能力，揭示领域知识、感知接地和推理方面的系统性不足。

**[Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users](joint_and_streamwise_distributed_mimo_satellite_communications_with_multi-antenn.md)**

:   提出面向多天线地面用户的分布式LEO卫星下行链路两种传输方案（联合传输 & 流式传输），通过基于统计CSI的WMMSE预编码设计和基于匈牙利算法的流-卫星关联策略，在无需卫星间相位同步的前提下实现了高频谱效率与低前传开销的灵活折中。

**[Lumosaic: Hyperspectral Video via Active Illumination and Coded-Exposure Pixels](lumosaic_hyperspectral_video_via_active_illumination_and_coded-exposure_pixels.md)**

:   提出Lumosaic主动高光谱视频系统，将12个窄带LED阵列与编码曝光像素（CEP）相机在微秒级同步，在每帧158个子帧内联合编码空间-时间-光谱信息，实现30fps VGA分辨率31通道（400–700nm）运动鲁棒高光谱视频重建，PSNR比被动快照系统高10+dB。

**[MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging](metaspectra_a_compact_broadband_metasurface_camera_for_snapshot_hyperspectral_im.md)**

:   MetaSpectra+ 提出超表面-折射透镜混合光学范式，通过双层超表面独立控制4通道色散/曝光/偏振，实现250nm宽带、17mm最短光程的快照式高光谱+HDR/偏振多功能成像，在KAUST基准上PSNR达33.31dB全面超越现有快照高光谱系统。

**[No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors](no_labels_no_look-ahead_unsupervised_online_video_stabilization_with_classical_p.md)**

:   提出无监督在线视频稳定框架 LightStab，通过经典三阶段管线（运动估计→运动传播→运动补偿）搭配多线程异步缓冲，在 5 个基准数据集上首次让在线方法全面媲美离线 SOTA，并发布首个包含可见光和红外的多模态无人机航拍稳定测试集 UAV-Test。

**[Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments](olbedo_an_albedo_and_shading_aerial_dataset_for_large-scale_outdoor_environments.md)**

:   Olbedo 提出首个大规模真实航拍反照率-着色分解数据集（5664 张 UAV 图像、4 种地貌、跨年多光照），通过物理逆渲染管线生成多视图一致的伪真值标注，证明合成预训练+Olbedo LoRA 微调可以显著提升室外反照率预测并支持重光照/材质编辑/场景变化分析等下游应用。

**[Are Pretrained Image Matchers Good Enough for SAR-Optical Satellite Registration?](pretrained_image_matchers_for_sar_optical_satellite_registration.md)**

:   本文在零样本设置下评估了24个预训练图像匹配器族在SAR-光学卫星配准上的表现，发现部署协议选择（几何模型、tile大小等）对精度的影响可达33倍，有时超过更换匹配器本身的效果。

**[RHO: Robust Holistic OSM-Based Metric Cross-View Geo-Localization](rho_robust_holistic_osm-based_metric_cross-view_geo-localization.md)**

:   提出首个面向恶劣天气和传感器噪声的OSM-based度量级跨视角定位基准CV-RHO（270万+ 图像），并设计双分支Pin-Pan架构RHO模型，结合全景去畸变（SUM）和位置-朝向融合（POF）机制，在多种退化条件下将定位性能提升高达20%。

**[SDF-Net: Structure-Aware Disentangled Feature Learning for Optical-SAR Ship Re-identification](sdfnet_structureaware_disentangled_feature_learnin.md)**

:   提出SDF-Net——物理引导的结构感知解耦特征学习网络，通过中间层梯度能量提取几何结构一致性(SCL)和终端层共享/模态专用特征解耦(DFL)+无参数加法融合，在HOSS-ReID上mAP达60.9%(+3.5% vs SOTA TransOSS)。
