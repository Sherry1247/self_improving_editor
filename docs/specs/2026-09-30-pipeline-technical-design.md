# 背景替换 Critic 闭环 — Pipeline 与技术设计

*日期：2026-09-30。前置：[`2026-09-30-bg-replacement-critic-novelty-and-plan.md`](./2026-09-30-bg-replacement-critic-novelty-and-plan.md)（创新点 A–E 的来源）、[`2026-09-10-critic-pipeline-architecture-design.md`](./2026-09-10-critic-pipeline-architecture-design.md)（`Evaluator` / `RefinementLoop` / `ModelRegistry` 接口）。本文把 A–E 全部落到模块上。*

---

## 0. 总览：四个层次

| 层 | 作用 | 对应创新点 |
|---|---|---|
| **L1 评估器 Evaluator** | (原图, 编辑图, 指令) → 分区域、分三分支的 critic 向量 | A（不变 / 目标 / 协变）、B（mask 自动区域 QA） |
| **L2 聚合 Aggregator** | critic 向量 → 门控 + 几何平均的总分 + 失败原因 | D（防 hacking） |
| **L3 闭环 Loop** | 生成 → 评估 → 诊断 → 动作 → 再评估 | E（诊断 → 动作路由） |
| **L4 跨样本记忆 Memory** | 学习「哪类失败用哪个动作」+ 成功改写案例检索 | E（跨样本自我改进） |
| **离线：Critic 验证** | 可控违例测试集 → 检验 L1 本身是否可靠，并给出 critic 可靠性权重 | C（以及 D 的权重来源） |

```
                    ┌──────────── 离线 ────────────┐
                    │ 违例测试集生成器 (C)          │
                    │  → critic AUROC / 校准         │──► 可靠性权重 w_i ──┐
                    └───────────────────────────────┘                    │
                                                                          ▼
(I_b, instr) ─► M0 EditSpec ─► M1 原图感知(缓存) ─┐                    M5 聚合
                                                 │                       │
                 ┌──────────── 闭环（每轮）──────┼─────────────────┐    │
                 │  M2 编辑器 ×N 候选             ▼                 │    │
                 │        └─► M3 编辑图感知 + 区域划分 S/B/C/G     │    │
                 │                 └─► M4 Critics: Gate/Keep/Follow/World ─┘
                 │                                   │
                 │   M6 诊断→动作路由 ◄── 总分+类型化问题
                 │     (prompt改写 / 调参 / 主体还原 / 局部inpaint / relight)
                 │                └─ 每个动作后重新评估，只接受提升
                 └──────── 停止：达到阈值 / 连续不提升 / 编辑预算耗尽 → 输出最优
                                   │
                                   ▼
                        M7 经验记忆：(问题类型, 编辑器, 动作) → Δscore
                        + 成功 prompt 改写案例库 → 反哺 M6
```

---

## 1. 模块详解

### M0 EditSpec 解析（把指令变成可检验的契约）
- **输入**：`labels.csv` 行（object, action, background）+ 目标背景 + 自由文本指令。
- **输出**（pydantic 校验的 JSON）：
  ```json
  {
    "subject": "dog", "subject_pose": "sit",
    "old_bg": ["river", "water"], "target_bg": "snowy field",
    "target_support": "snow ground",
    "invariants": ["identity", "pose", "silhouette", "texture"],
    "covariants": ["support_contact", "contact_shadow", "light_direction",
                   "color_temperature", "reflection_removed", "scale"],
    "forbidden": ["extra_subject", "text", "subject_cropped"]
  }
  ```
- **技术**：规则模板生成主体部分（来自标签，可靠）；用本地 LLM（Qwen2.5-7B-Instruct，JSON 约束输出）补 `target_support`，并判断哪些协变项适用。例如目标背景仍是水面时，`reflection_removed` 不适用。
- **为什么需要**：后面所有 region QA 问题、门控阈值、动作选择都由它实例化，实现「instruction-conditioned evaluation」。

### M1 原图感知（每张图只跑一次，结果缓存成 npz/json）
| 输出 | 技术 |
|---|---|
| 主体 mask `M_b` | **SAM 3**（文本提示 "dog"/"person"）；备选 Grounding DINO + SAM 2 |
| 主体计数 | Grounding DINO（`IDEA-Research/grounding-dino-base`，transformers 原生），与 SAM 3 实例数交叉验证 |
| 深度 `D_b` | Depth Anything V2（相对深度） |
| 身份特征 | DINOv2 ViT-B/14：主体 masked crop 的 CLS 向量 + patch 特征 |
| 姿态 | 人：YOLO11-pose / RTMPose 关键点；狗：轮廓 + DINOv2 patch 对应（不训练动物姿态模型） |
| 光照统计 | 主体区 Lab 均值/方差、亮度梯度主方向（供 K3 归一化和 W3 使用） |

### M2 编辑器（统一 `Editor` 接口，每轮生成 N 个候选）
| 编辑器 | 角色 | 可调参数（动作空间） |
|---|---|---|
| InstructPix2Pix（现有） | 弱基线 | `guidance_scale`、`image_guidance_scale`、seed |
| **FLUX.1 Kontext-dev**（或 Qwen-Image-Edit / Step1X-Edit，选一个） | 主力 | `guidance`、steps、seed |
| **Compositing 基线**：SAM 3 抠主体 → FLUX.1-Fill / SDXL-Inpaint 对背景区重绘 | 「主体完美保留、物理通常差」的对照 | 背景 prompt、seed、mask 膨胀量 |

`Editor.action_space()` 由各实现自己声明，路由器（M6）只从中选择动作，因此换编辑器不需要改路由逻辑。

### M3 编辑图感知 + 区域划分（创新点 B 的基础）
1. SAM 3 → `M_a`；DINOv2 → 身份特征；Depth Anything V2 → `D_a`。
2. **对齐**：用 mask 矩（质心、尺度、主轴）估计相似变换 T，把 `M_b` 对齐到 `M_a`，允许编辑器有轻微位移。
3. **区域划分**（全部由 mask 推出，不需要人工标注）：
   - **S（主体）** = `M_a`
   - **B（边界带）** = `dilate(M_a, k) − erode(M_a, k)`，k ≈ 主体短边的 3%
   - **C（接触带）** = 主体 mask 最低处 10% 高度的像素，向下延伸主体高度的 15%、左右各外扩 20%
   - **G（背景）** = `¬dilate(M_a, 2k)`
4. **主体挖除后的背景图**：用 LaMa 把主体区补掉，得到纯背景图 `I_G`，供语义打分。如果嫌依赖太多，简化版用模糊填充。

### M4 Critics（每个返回 `CriticResult{score∈[0,1], issues: list[IssueType], evidence, region, is_catastrophic}`）

**Gate（硬约束，任一失败即判定为灾难性）**
| ID | 检验 | 技术 / 公式 |
|---|---|---|
| G1 | 主体存在且只有 1 个 | GDINO + SAM 3 计数，两者都为 1 |
| G2 | 身份未丢 | cos(DINOv2(S_b), DINOv2(S_a)) ≥ τ_id |
| G3 | 背景确实换了（防「不改图」hack） | G 区 LPIPS(I_b, I_a) ≥ τ_bg |

**Keep 分支（不变量，区域 S）— 创新点 A 的「不该变」**
| ID | 检验 | 技术 |
|---|---|---|
| K1 | 轮廓 | 对齐后 mask IoU |
| K2 | 姿态 | 人：关键点 PCK@0.1；狗：DINOv2 patch 互近邻一致率 |
| K3 | **光照归一化后的外观** | 在 Lab 空间对 S 区做低频亮度/色偏的归一化（逐通道均值方差匹配），再算 masked LPIPS。允许光照变化，不允许纹理或结构变化 |
| K4 | 局部纹理 | DINOv2 patch 特征逐 patch 余弦均值 |

**Follow 分支（目标变化，区域 G）**
| ID | 检验 | 技术 |
|---|---|---|
| F1 | 背景是目标场景 | SigLIP（so400m）对 `I_G` 做**候选背景集合上的 softmax 分类**（目标背景 + 其余 5 种 + 旧背景），比单独看余弦更稳 |
| F2 | 旧背景已移除 | SAM 3 对 `old_bg` 概念（water/river）在 G 区的面积占比 ≈ 0 |
| F3 | VLM 确认 | 「Is the background a {target_bg}?」→ P(Yes) |

**World 分支（协变量，区域 C/B/全局）— 创新点 A 的「必须跟着变」**
| ID | 检验 | 启发式信号 | VLM region QA |
|---|---|---|---|
| W1 | 支撑 / 不悬空 | SAM 3 分割 `target_support`，检查 C 区下方是否有支撑面；深度连续性：\|D(主体底部) − D(C 区地面)\| 小 | 「Is the {subject} resting on the {target_support}, not floating?」 |
| W2 | 接触阴影 | C 区亮度相对外圈地面环带的下降比 | 「Is there a contact shadow under the {subject}?」 |
| W3 | 光照方向 / 色温 | 背景与主体高光区的 Lab b* 差（色温一致）；主体亮度梯度方向 vs 背景阴影方向 | 「Is the {subject} lit from the same direction as the scene?」 |
| W4 | 尺度 / 透视 | 主体像素高度 vs 接触点深度（与地平线位置的一致性） | 「Is the {subject}'s size plausible for its position?」 |
| W5 | 边界伪影 | B 区内外侧颜色差 + 高频边纹（OpenCV） | 「Is there a halo or cut-out edge around the {subject}?」 |
| W6 | 残留倒影 | 当 `reflection_removed` 适用时检查 C 区下方 | 「Is there a reflection of the {subject} below it?」 |

**Region QA 的实现细节（创新点 B）**
- 问题由 `EditSpec.covariants × 区域模板` 自动生成，不需要人工写。
- VLM：**Qwen2.5-VL-7B-Instruct**（或 Qwen3-VL 同尺寸），本地运行。
- 输入形式：区域裁剪图 + 全图（区域用红框标出），两张图一起送入。
- 打分：取「Yes」token 的概率 **P(Yes)** 作为连续分数，而不是让模型直接输出 1–10 分。再采样 3 次，记录方差，供可靠性分析使用。
- 对照（E5）：同样的问题改成全图提问、不做区域绑定。

### M5 聚合（创新点 D）
```
if any(gate.is_catastrophic): score = 0, reason = 失败的 gate
else:
    branch_b = Σ_i w_i · s_i / Σ_i w_i         # w_i = critic 可靠性（来自离线 AUROC / 人工一致性）
    score = (Keep · Follow · World)^(1/3)       # 几何平均：任一分支低就会拉低总分，不能互相补偿
return score, 分项向量, issues
```
- **防 hacking 监控**：保留一个不参与优化的评估器集合 H（EditReward 或 EditScore + 人工抽检）。每轮记录 Δscore_loop 与 Δscore_H；两者背离就说明闭环在钻空子。
- 对照组：现有的扁平加权和。

### M6 诊断 → 动作路由（创新点 E）
**问题类型（枚举）→ 候选动作**
| IssueType | 来源 | 候选动作（按先验优先级） |
|---|---|---|
| EXTRA_SUBJECT, IDENTITY_LOSS, SUBJECT_DRIFT | G1/G2/K* | a2 提高原图保持强度 → a3 主体还原 |
| BG_UNCHANGED, BG_WRONG, OLD_BG_RESIDUE | G3/F* | a1 prompt 改写 → a2 降低原图保持强度 / 换 seed |
| FLOATING, MISSING_SHADOW | W1/W2 | a4 接触带局部 inpaint |
| LIGHT_MISMATCH | W3 | a6 主体 relight → a3 的 harmonization |
| SCALE_WRONG | W4 | a1 prompt 改写（写明尺度）→ 换 seed |
| HALO | W5 | a5 边界带局部 inpaint |
| REFLECTION_RESIDUE | W6 | a4 / a5 局部 inpaint |

**动作实现**
- **a1 prompt 改写**：LLM 接收「当前 prompt + 类型化问题 + 记忆库中检索到的成功改写案例」，输出新 prompt。长度有上限，每次是重写而不是追加句子，以修复现有代码中 prompt 无限变长的问题。
- **a2 调参**：在 `Editor.action_space()` 内按问题方向调整参数。
- **a3 主体还原**：把原图主体（对齐后的 `M_b`）贴回编辑图，再做协调。廉价版用 Lab 色彩迁移（Reinhard）；完整版用 IC-Light 做前景条件 relight。
- **a4 / a5 局部 inpaint**：FLUX.1-Fill 或 SDXL-Inpaint，只在 C 区或 B 区内修改，prompt 由 EditSpec 生成，例如「soft contact shadow on snow under the dog's paws」。
- **a6 relight**：IC-Light，以背景作为光照条件。
- **规则**：先修 gate，再修分数最低的分支；局部动作只改自己的区域，不会破坏其他已经通过的部分。每个动作执行后重新评估，**只接受提升的结果**（accept-if-improves）。

### M7 经验记忆（跨样本自我改进）
- **动作统计表**：`(IssueType, editor, action) → {n, mean Δscore}`。路由选择动作时用 UCB 或 Thompson sampling，代替固定的先验优先级。
- **改写案例库**：保存成功的 (EditSpec, 问题, 旧 prompt → 新 prompt, Δscore)，按 EditSpec 相似度检索，作为 a1 的 few-shot 示例。
- **评估方式**：按数据集顺序画学习曲线，看「到达阈值所需的编辑调用次数」是否随处理过的样本增多而下降。

### 离线：违例测试集生成器（创新点 C）
- **巧妙之处**：直接用**真实照片作为正样本**，因为真实照片本来就物理正确；再对真实照片做受控扰动得到负样本。这样正负样本只差一种违例，干净、可控。
- **流程**：SAM 3 抠出主体 → 用 LaMa 补出空背景 → 施加扰动 → 合成回去。
| 违例 | 扰动方式 |
|---|---|
| 无阴影 | 抠出后直接贴回空背景（原阴影已被 LaMa 抹掉） |
| 阴影方向错误 | 用 mask 投影生成假阴影，方向与场景光照相反（正样本：方向一致） |
| 悬空 / 下沉 | 主体上移或下移 Δy 像素 |
| 尺度错误 | 缩放 ×0.6 / ×1.5，接触点保持不动 |
| 边缘光晕 | 用膨胀后的 mask 抠取（带上原背景像素），再贴到别的背景上 |
| 色温不一致 | 只对主体做色温偏移 |
| 残留倒影 | 把主体垂直翻转，以低透明度叠加在主体下方 |
| 主体漂移 | 用 IP2P 对主体区做轻微编辑 |
- **输出**：每个 critic 在每类违例上的 AUROC；VLM 的重复采样方差；→ 给 M5 的可靠性权重 `w_i`。

---

## 2. 运行方式（工程层面）

**按阶段批处理，而不是逐样本交错执行。** 主力编辑器（约 12B）和 VLM（7B）在同一张 40GB 卡上很难常驻，所以每一轮按阶段执行：
```
Round r:
  Stage E: 加载编辑器 → 对所有「未完成」样本生成 N 个候选 → 卸载
  Stage P: 加载 SAM3/GDINO/DINOv2/Depth/SigLIP → 感知 + 区域划分 → 卸载
  Stage V: 加载 VLM → 批量 region QA → 卸载
  Stage A: CPU：聚合 + 路由 + 记忆更新 → 写出下一轮的任务表
```
- 每个阶段的输入输出都落盘（`results/<run_id>/round_r/<stage>/…`），可以断点续跑。HTCondor 上一个阶段对应一个 job，适合 CHTC。
- `ModelRegistry` 按 `(kind, model_id, device)` 作为缓存键，并提供 `unload(kind)`。
- 每个输出 JSON 顶层都带 `mock`、`model_versions`、`seed`、`edit_budget_used`。
- **编辑预算**：比较不同方法时统一以「编辑器调用次数」为预算，保证 E4 的公平。

**依赖与环境**
- 现有环境：torch 2.11、diffusers 0.37、transformers 5.5、ultralytics 8.4。
- GDINO、DINOv2、SigLIP、Depth Anything V2、Qwen-VL 都优先用 **transformers 原生实现**，避免 GDINO 原始仓库的 CUDA 编译。
- SAM 3、LaMa、IC-Light 依赖较老或自带代码，放进**独立的 conda env / 容器**，作为子进程阶段运行（阶段式执行正好允许这样做）。
- 新增 pip 依赖：`lpips`、`pydantic`、`scikit-learn`（AUROC）、`vllm`（可选，加速 VLM 批量问答）。
- FLUX.1-dev 系列权重是非商用许可，研究用途没有问题，但需要在 HF 上接受协议。

---

## 3. 目录结构增量（在 Phase 2 spec 基础上）
```
src/
  spec/edit_spec.py            # M0 EditSpec + 模板 + LLM 补全
  perception/                  # M1/M3：sam3.py, gdino.py, depth.py, dino.py, pose.py, lama.py
  regions/partition.py         # S/B/C/G + 对齐
  critics/
    gate.py  keep.py  follow.py  world.py
    region_qa.py               # 问题模板 + VLM P(Yes) 打分 + 采样方差
  scoring/aggregation.py       # 门控 + 几何平均 + 可靠性权重
  refinement/
    issues.py                  # IssueType 枚举
    router.py                  # M6 诊断 → 动作
    actions/                   # prompt_rewrite.py, param_adjust.py, subject_restore.py, local_inpaint.py, relight.py
    memory.py                  # M7 动作统计 + 案例检索
  editors/  ip2p.py  kontext.py  compositing.py
  pipelines/  evaluator.py  refinement_loop.py  staged_runner.py
validation/
  violation_suite.py           # C：真实照片 → 受控违例
  critic_auroc.py
experiments/  e1_critic_auroc.py … e6_editor_tradeoff.py
```

---

## 4. 最小可行版本（先跑通，再加东西）
1. **MVP（W1–W3）**：M0 只用模板；M1/M3 用 SAM 3 + DINOv2 + LPIPS；critic 只做 G1–G3 + K1/K3 + F1；M5 先用门控 + 几何平均；M6 只有 a1/a2。→ 闭环能跑。
2. **+World（W4–W5）**：接入 VLM region QA（W1、W2、W3、W5）+ 深度。
3. **+C（W6）**：违例测试集 → AUROC → 权重。
4. **+E 完整（W7）**：a3–a6 局部动作 + M7 记忆。

---

## 5. 实现状态（2026-09-30，分支 `feat/bg-critic-mvp`）

**已实现**：M0 EditSpec（模板版）· M1/M3 感知（GDINO + SAM2 + DINOv2 + SigLIP + Depth Anything V2，原图缓存）· 区域 S/B/C/G · Gate 三项 · Keep（silhouette / 光照归一化 appearance / texture）· Follow（SigLIP 背景分类 / 旧背景残留）· World 启发式（support / contact_shadow / light_harmony / halo）· 门控几何平均聚合（加权和作为基线同时记录）· 编辑器 IP2P（保持比例）+ compositing 基线 · 路由 a1（clause）/ a2（参数）/ reseed · M7 动作记忆（UCB，跨任务持久化）· 违例测试集 + AUROC 脚本。

**与本文设计的差异**：主体移除用 OpenCV Telea（不是 LaMa）；姿态 critic（K2）未实现；W3 用高光色度（max-RGB 估计光源色）代替光照方向；W4 尺度、W6 倒影留给 VLM 版本。

**下一步**：VLM region QA（Qwen2.5-VL）· 局部修复动作 a3–a6（主体还原 / 接触带 inpaint / relight）· FLUX Kontext 编辑器 · SAM 3 替换 GDINO+SAM2。
