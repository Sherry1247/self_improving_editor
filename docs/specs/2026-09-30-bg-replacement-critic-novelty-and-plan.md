# 背景替换 Critic 闭环：新颖性分析 · Pipeline 框架 · 执行计划

*日期：2026-09-30。前置文档：[`PHASE1_AUDIT.md`](../PHASE1_AUDIT.md)、[`2026-09-10-critic-pipeline-architecture-design.md`](./2026-09-10-critic-pipeline-architecture-design.md)。本文接在 Phase 2 架构 spec 之后，补上它明确推迟的部分（聚合方式、instruction conditioning、物理 critic 的范围），并给出研究定位。*

---

## 1. 这个方向是不是很多人做过？

**结论：通用骨架（检测 + VLM + 物理 critic → 加权求和 → prompt refinement 闭环）已经被做得很多了，单靠这个骨架不构成创新。** 下面按模块列出已有工作：

| 我们的模块 | 已有工作（代表） | 重合程度 |
|---|---|---|
| 编辑 reward / critic 模型 | VIEScore；EditReward（ICLR 2026）；EditScore（ICLR 2026，用于在线 RL） | 高：「给编辑结果打分」本身已经是成熟方向 |
| 把指令拆成可验证的分项 | Edit-R1 / verifier-based RRM（CVPR 2026）：把指令拆成 **Keep / Follow / Quality** 三类原则，再用 CoT 逐条验证 | **很高**：几乎就是我们 track 2 的「检测 / 语义 / 世界」三分支 |
| 物理合理性评估 | PICABench + PICAEval（ICLR 2026）：光学（阴影/反射/折射/光源）、力学（形变/因果）、状态变化；**region-bound 是/否问答**，区域靠人工标注 | 高：「world critic」的评测方法已经被提出 |
| 迭代自我改进 + prompt 改写 | Idea2Img（ECCV 2024）、PRISM（2026）、Iterative Refinement for Compositional Generation（2026）、各种 agentic editing 工作 | 高：「critic → 改 prompt → 再生成」是标准做法 |
| 用检测器做计数 / 空间验证 | GenEval、T2I-CompBench 一类 | 高：计数 critic 是常规工具 |
| 背景替换 + 光照协调 | Relightful Harmonization（Adobe）、IC-Light、AnyPortal（ICCV 2025）、MultiShadow（2026 阴影生成） | 中：它们解决的是**生成**，不是**评估 + 闭环** |

所以，如果论文标题是「a multi-critic self-improving image editor」，审稿人会直接拿 Edit-R1 + PICABench + Idea2Img 来对比，认为是组装。

## 2. 还剩哪些微弱但站得住的创新点？

关键是**把题目收窄到「背景替换」，并利用这个任务独有的结构**。通用编辑 critic 不知道哪些区域「该变」、哪些「不该变」、哪些「必须跟着变」；而背景替换中这三者都能从主体 mask 自动推出。

### 创新点 A（主线）：Invariant–Covariant 契约 critic
已有 critic 只有两种视角：**Keep**（不该变的有没有变）和 **Follow**（要求变的有没有变）。背景替换里还有第三类：**必须跟着变（co-change）**。

- **Invariant（不变）**：主体身份、姿态、轮廓、材质。
- **Target change（目标变化）**：背景语义变成目标场景，旧背景消失。
- **Covariant（协变）**：主体与新环境的物理耦合必须随背景一起更新——支撑面（原来坐在河边石头上，换到雪地后必须坐在雪面上，而不是悬空）、接触阴影、光照方向与色温、倒影（离开水面后原有倒影必须消失）、透视/尺度。

Edit-R1 的 Keep/Follow 会把「主体光照变了」判成违反 Keep；而在背景替换中，**主体的光照应该变**。这个矛盾就是可以写的点：*preservation 要在光照归一化之后测，physics 要测「是否正确地变了」*。

### 创新点 B：从 mask 自动生成 region-bound 物理问答（去掉人工标注）
PICAEval 的 region-bound QA 效果好（与人的相关约 0.95），但区域靠**人工标注**。背景替换中区域可以从 SAM mask 自动划分：
主体区 S / 边界带 B（mask 膨胀−腐蚀）/ 接触区 C（mask 底部膨胀条带）/ 背景区 G。
每个区域绑定一组模板化的是/否问题，由 EditSpec 实例化。贡献是：**自动的 region-bound 物理评估，并与人工区域版本做对比**。

### 创新点 C：可控的物理违例测试集，用来检验 critic 本身
用合成扰动造「已知错误」：只贴主体不加阴影、阴影方向翻转、主体悬空 / 下沉 N 像素、尺度与深度不符、边缘光晕、残留旧倒影、主体颜色漂移。每类给出正负样本，测各 critic 的 AUROC。这能回答一个很实际的问题：*CLIP / 单次 VLM 打分 / EditReward / 我们的 region critic，到底谁能看出物理错误？* 成本低，结果清楚，单独就够写一篇 workshop 论文。

### 创新点 D：闭环中的 reward hacking 分析 + 门控聚合
自我改进闭环会去优化 critic 的漏洞，例如：
- 编辑器几乎不改图 → preservation 满分，加权和仍然很高；
- 主体被毁，但 CLIP / VLM 分数高，把总分撑起来。

做法：**双向硬门控**（主体保留 ≥ τ₁ 且背景确实变化 ≥ τ₂）+ 门控之后取几何平均；再用 **held-out critic**（不参与优化的评估器 + 人工）来衡量「优化 critic A 时，critic B 的分数是否同步上升」。这能直接回应 Phase 1 审计提出的 Experiment 4。

### 创新点 E（工程向，加分项）：诊断 → 动作路由，而不只是改 prompt
已有 loop 基本只做 prompt 改写。背景替换的失败类型可以对应到具体修复动作：

| 诊断 | 动作 |
|---|---|
| 背景没变 / 变得不够 | 改 prompt，降低 `image_guidance_scale` |
| 主体漂移 | 提高 `image_guidance_scale`；或 fallback：用 mask 把原主体贴回去，再做 harmonization |
| 接触区缺阴影 / 悬空 | 只对接触区做局部 inpaint（prompt 写明 contact shadow） |
| 光照不一致 | 对主体做 relight（IC-Light 类工具）或 harmonization |
| 残留旧倒影 / 光晕 | 对边界带 B 做局部 inpaint |

再加一个**跨样本经验记忆**：记录（失败类型, 动作, Δscore），学出哪类失败用哪个动作最有效（简单的 bandit 即可）。这样「self-improving」就不只发生在单张图内，也发生在整个数据集上。

### 建议的定位
**主线 = A + B + C，聚合与防 hacking = D，E 作为闭环部分的 demo。** 一句话版本：

> 背景替换需要的不只是「保留主体」，还要「让主体与新环境正确耦合」。我们提出 invariant–covariant 的区域化 critic，从主体 mask 自动生成，并用可控违例集验证它比全局 critic / 通用 reward model 更能识别物理错误；在门控聚合下，它驱动的闭环比只改 prompt 的闭环更不容易被 hack。

### 要诚实面对的弱点
- 「physics」实际是**图像层面的物理一致性启发式**（支撑、阴影、光照、尺度），不是物理仿真，命名时要克制（审计 §4 也指出了这一点）。
- 只做人和狗、坐和站两种姿态：范围窄，但对 workshop / 课程项目是优点（可控）。
- VLM 的打分方差需要测（重复采样），否则无法回答「这个分数是否可复现」。

---

## 3. Pipeline 框架

两条 track 统一成一条，以 track 2 的三分支为组织结构，但**保留 before 图作为参照**。只看编辑后的图无法判断主体是否被保留，所以 track 2 的「仅输入 edited image」对本任务不够。

```
(I_before, instruction)
      │
      ▼
[0] EditSpec 解析 ── 模板 + LLM ──► {subject, target_bg, old_bg,
      │                               invariants, covariants, forbidden}
      ▼
[1] Before 感知（每张图只跑一次，缓存）
      SAM3（文本提示）或 GDINO+SAM2 → 主体 mask M_b
      Depth Anything V2 → 深度 D_b
      主体特征：DINOv2 masked-crop embedding；人体加 2D 关键点
      │
      ▼
[2] Editor 生成 N 个候选（best-of-N）
      主力：开源 instruction editor（FLUX.1 Kontext-dev / Qwen-Image-Edit / Step1X-Edit 选一）
      基线1：InstructPix2Pix（现有）
      基线2：Compositing = 抠主体 + 背景 inpaint（主体完美保留，物理通常差）
      │
      ▼
[3] After 感知 + 区域划分
      M_a, D_a → 区域 S（主体）/ B（边界带）/ C（接触带）/ G（背景）
      │
      ▼
[4] Critic 三分支（全部读同一份缓存，不重复跑模型）
  ┌ Gate（硬约束）
  │   · 主体存在且数量 = 1（GDINO 计数，用 SAM3 交叉验证）→ 防止幻觉出多余主体
  │   · 身份：DINOv2 masked-crop 余弦 ≥ τ_id
  │   · 背景确实变了：G 区 LPIPS ≥ τ_bg（防止「不改图」hack）
  ├ Detection / Keep 分支（区域 S）
  │   · 轮廓：对齐后的 mask IoU；姿态：关键点 PCK（人）/ 轮廓形状（狗）
  │   · 外观：亮度/色温归一化之后的 masked LPIPS（允许光照变化，不允许纹理变化）
  ├ Semantic / Follow 分支（区域 G + 全局）
  │   · 主体挖掉后，G 区与 target_bg 的 SigLIP/CLIP 相似度
  │   · 旧背景消失：对 old_bg 概念做检测（如 river/water 不应再出现）
  │   · VLM 是/否问答：「背景是否是 {target_bg}？」
  └ World / Covariant 分支（区域 C、B、全局）
      · 支撑：C 区下方是否有可支撑表面（分割 + 深度连续性：主体底部深度 ≈ 地面深度）
      · 接触阴影：C 区亮度是否下降（启发式）+ VLM region QA
      · 光照一致：主体亮侧与背景光源方向是否一致（VLM QA + 简单着色统计）
      · 尺度/透视：主体高度与所在深度的关系
      · 边界伪影：B 区光晕 / 颜色边纹检测
      · 残留：离开水面后是否仍有倒影（VLM region QA）
      │      ↑ 每个问题由 EditSpec 模板生成，并绑定到对应区域的裁剪图上
      ▼
[5] 聚合：Gate 不通过 → score = 0，并带上失败原因；
          通过 → 各分支几何平均（可加权）；同时保留完整分项向量用于诊断
      │
      ▼
[6] 诊断 → 动作路由（见 §2-E）：改 prompt / 调参 / 局部修复工具
      │
      ▼
[7] 经验记忆：(失败类型, 动作, Δscore) → 更新路由策略（跨样本自我改进）
      │
      └──► 回到 [2]，直到：分数 ≥ 阈值 / 连续不提升 / 达到最大迭代次数；输出历史最优候选
```

**与 Phase 2 spec 的对应**：沿用 `Evaluator` / `RefinementLoop` / `ModelRegistry` / `CriticResult(is_catastrophic)` / `Aggregator` 这些接口；新增 `EditSpec`、`RegionPartition`、`ActionRouter`、`ExperienceMemory` 四个模块。`ObjectRole` 字段在这里落地：主体 = PRIMARY_SUBJECT，旧背景概念 = BACKGROUND_CONTEXT（应当被移除）。

**模型取舍（CHTC GPU）**：VLM 用本地开源模型（如 Qwen2.5-VL-7B / Qwen3-VL），不依赖 API；每个问题采样 3 次取多数，同时记录方差。SAM3 支持文本提示分割，可以替代 GDINO+SAM2；GDINO 仍保留给计数 critic 做交叉验证。

---

## 4. 实验设计

| 编号 | 问题 | 做法 | 指标 |
|---|---|---|---|
| E1 | critic 能否识别物理错误？ | 违例测试集（§2-C，约 8 类 × 每类 100+ 对） | 每个 critic、每类违例的 AUROC；对比全局 CLIP、单次 VLM 总分、EditReward/EditScore、我们的 region critic |
| E2 | 与人的一致性 | 约 150–300 个真实编辑结果，2 人标注（成对偏好 + 分项是/否） | Spearman、成对准确率、Cohen's κ |
| E3 | 聚合方式与 hacking | 加权和 vs 门控 + 几何平均；在闭环中优化，用 held-out 评估器测量 | held-out 分数随迭代的变化曲线、hack 案例数 |
| E4 | 闭环是否有效 | 一次生成 / best-of-N / 只改 prompt 的闭环 / 诊断路由闭环，**相同编辑调用次数** | held-out 分数 + 人工胜率 |
| E5 | 消融 | 去掉每个分支或每个区域；自动区域 vs 全图 QA | 同 E1/E4 |
| E6 | 编辑器对比 | IP2P / 强 editor / compositing 基线 | 展示「保留–物理」的权衡：compositing 保留满分但物理差 |

**数据**：现有 15 张图太少。扩充到约 120–200 张（人/狗 × 坐/站 × 多种源背景），目标背景 5–6 种，并覆盖不同的支撑面类型（草地、雪地、沙滩、城市路面、室内地板、水边）。支撑面的变化正是协变 critic 要检验的东西。

---

## 5. 执行计划（10 周，从 10/1 开始）

| 周 | 时间 | 交付 |
|---|---|---|
| W1 | 10/1–10/7 | 落地 Phase 2 重构：`Evaluator` + `RefinementLoop` + `ModelRegistry`；真实 GDINO/SAM2(3)/CLIP 跑通，`mock` 标志写进每个输出 JSON；清理当前工作区未提交的改动 |
| W2 | 10/8–10/14 | 接入一个强 editor + compositing 基线；数据扩充到 ≥100 张；`EditSpec` 模板 |
| W3 | 10/15–10/21 | `RegionPartition`（S/B/C/G）+ Gate critic + Keep 分支（DINOv2 身份、对齐 IoU、光照归一化 LPIPS） |
| W4 | 10/22–10/28 | Follow 分支 + 本地 VLM 接入（多次采样、记录方差）；region QA 模板库 |
| W5 | 10/29–11/4 | World 分支（深度支撑、接触阴影、光照、尺度、边界伪影、残留倒影） |
| W6 | 11/5–11/11 | 违例测试集生成器 + **E1**（第一个硬结果，决定后续叙事） |
| W7 | 11/12–11/18 | 门控聚合 + `ActionRouter` + `ExperienceMemory`；闭环在 CHTC 上批量运行 |
| W8 | 11/19–11/25 | 人工标注（E2）；**E3** hacking 分析 |
| W9 | 11/26–12/2 | **E4/E5/E6**，在相同计算预算下对比 |
| W10 | 12/3–12/9 | 图表、失败案例集、报告 / 论文初稿 |

**里程碑与止损**：
- W6 结束时如果 E1 显示 region critic 并不比单次 VLM 总分好 → 叙事转为「评测分析」（哪些物理错误谁都看不出），闭环部分缩小。
- W7 如果强 editor 在 CHTC 上显存或时间不够 → 主力退回 IP2P + compositing 两个基线，结论仍然成立（权衡的两端）。

**分工建议**（我们负责背景替换 + 物理不变/协变）：critic 与违例测试集（A/B/C）是我们的核心产出；聚合与闭环（D/E）和另一条 track 共用接口，可以协作。

---

## 参考

- PICABench / PICAEval（ICLR 2026）：https://en.papernotes.org/ICLR2026/image_generation/picabench_how_far_are_we_from_physical_realistic_image_editing/
- Edit-R1，verifier-based RL for editing（CVPR 2026）：https://en.papernotes.org/CVPR2026/image_generation/leveraging_verifier-based_reinforcement_learning_in_image_editing/
- EditReward（ICLR 2026）：https://github.com/TIGER-AI-Lab/EditReward
- EditScore（ICLR 2026）：https://github.com/VectorSpaceLab/EditScore
- Idea2Img（ECCV 2024）：https://idea2img.github.io/
- PRISM（2026）：https://arxiv.org/html/2607.24353v1
- Iterative Refinement Improves Compositional Image Generation：https://arxiv.org/html/2601.15286v1
- Relightful Harmonization：https://arxiv.org/html/2312.06886
- AnyPortal（ICCV 2025）：https://openaccess.thecvf.com/content/ICCV2025/papers/Gao_AnyPortal_Zero-Shot_Consistent_Video_Background_Replacement_ICCV_2025_paper.pdf
- MultiShadow：https://arxiv.org/html/2603.02743
