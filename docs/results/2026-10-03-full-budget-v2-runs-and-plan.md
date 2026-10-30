# Full-budget 三编辑器对比（critics v2）· 结论 · 下一步计划

*日期：2026-10-03。数据：CHTC 上的 `ip2p_v2`、`comp_v2`、`qwen` 三个 run，各 90 个任务（15 张图 × 6 个目标背景），
每个任务都花满 12 次编辑（3 轮 × 4 个 seed，`loop.full_budget`），critics v2 + `softmin2` world 聚合。
统计脚本：`experiments/analyze_runs.py`（直接读 `runs/<name>/*/report.json`）。
前置文档：[first CHTC runs](./2026-10-01-first-chtc-runs.md)、[critics v2](./2026-10-02-critics-v2.md)、
[新颖性与计划](../specs/2026-09-30-bg-replacement-critic-novelty-and-plan.md)。*

> 注意：`runs/*/summary.csv` 现在只有 6 行（`collect.sh` 解包时每个 job 的 summary.csv 互相覆盖了），
> 本文所有数字都是从 270 个 `report.json` 重新算的。需要修 `collect.sh`（见 §5）。

---

## 1. 主要数字

| | ip2p_v2 | comp_v2 | qwen |
|---|---|---|---|
| gate 通过：一次生成 → 第 0 轮 best-of-4 → 12 次后 | 13 → 19 → **58** | 89 → 90 → 90 | 89 → 90 → 90 |
| 严格成功（gate + SigLIP 与 VLM 都认可目标背景） | 33 / 90 | – | – |
| 总分 中位数 / 均值（12 次后） | 0.56 / 0.41 | **0.93 / 0.92** | 0.88 / 0.86 |
| 总分：一次生成 → best-of-4 → loop（均值） | 0.07 → 0.13 → 0.41 | 0.83 → 0.89 → 0.92 | 0.79 → 0.83 → 0.86 |
| loop 相对第 0 轮的平均提升 | **+0.28** | +0.028 | +0.024 |
| 第 0 轮 4 个 seed 的分数极差（中位数） | 0（多数全是 0） | 0.105 | 0.072 |
| keep / follow / world 中位数 | 0.83 / 0.57 / 0.59 | **0.99** / 0.99 / **0.85** | 0.84 / **1.00** / 0.82 |
| 两两胜负（按 critic 总分） | | comp > ip2p 90/90 | qwen > comp **15/90**，qwen > ip2p 87/90 |

按主体：ip2p 对狗几乎全失败（狗中位数 0.22，水中狗 0.14），对人 0.59；comp 和 qwen 对人和狗都稳定。
按目标：ip2p 的 snow 中位数为 0（9/15 被 gate），它把 "snow" 理解成整体调冷色。

## 2. 三个关键发现

### 发现 1：critic 偏爱「复制粘贴」——comp 赢 qwen 75/90，但从图上看 qwen 更合理

逐 critic 对比 qwen vs comp（90 个任务，最佳结果）：

| critic | qwen 胜 | comp 胜 | 说明 |
|---|---|---|---|
| appearance | 0 | **90** | comp 的主体逐像素不变，距离 = 0；qwen 光照归一化后距离仍约 10 |
| silhouette | 25 | 65 | qwen 有时改姿态（16 个任务 IoU < 0.85，如趴着的狗变成坐着）——这部分是**真问题** |
| texture | 0 | 22 | |
| light_harmony | 20 | **46** | ⚠ comp 保留的是**原图光照**，却被判为「光照更协调」 |
| bg_semantic / vlm_background | 76 / 82 | 14 / 6 | qwen 的背景更像目标 |
| vlm_surface / integration / lighting | 45 / 42 / 42 | 13 / 23 / 17 | qwen 的物理耦合更好 |

只看 follow × world（去掉 keep），qwen 与 comp 打平（45 / 90）；comp 的领先**全部来自 keep 分支**。
有 23 个任务 comp 的 world 分低于 qwen，但总分仍然更高。

这正好和创新点 A 冲突：背景替换中**主体光照本来就应该变**（协变），但现在的 `appearance` 和 `light_harmony`
都是「相对于原图」测的，实际上变成了「越像原图越好」的保留指标。原因：

* `appearance` 只去掉了全局色偏，qwen 重新打光后的局部明暗（侧光、阴影面）仍算漂移；
* `light_harmony` 以原照片为参照，比较的是「主体和背景的光照关系是否与原图一致」——comp 的主体没变、背景是 SDXL
  生成的高光，碰巧也通过。它没有测「主体是否被新场景的光照照亮」。

**这是目前最有价值的结论，也是论文的核心论据候选**：保留型 critic（包括我们的 v2，很可能也包括 EditReward / VIEScore
这类通用 reward）系统性地偏爱 compositing，把物理上没有耦合的结果排在前面。要成立，必须先做人工标注（E2）证明人更喜欢 qwen。

### 发现 2：loop 只对弱编辑器有用；对强编辑器 ≈ best-of-N

* **ip2p**：gate 通过数 19 → 58，平均 +0.28。几乎全部来自 `edit_more`（降低 `image_guidance_scale`）：73 次中 36 次被接受，
  平均 Δ = +0.23。prompt 子句类动作基本有害：`clause_contact` Δ −0.28、`clause_natural` −0.32、`clause_light` −0.25、
  `clause_preserve` −0.10。→ **对 ip2p 有效的是「诊断 → 参数」，不是「诊断 → 改 prompt」。**
* **comp / qwen**：loop 的提升（+0.024 ~ +0.028）小于同一 prompt 下 4 个 seed 之间的极差（0.07 ~ 0.10）。
  所有路由动作的平均 Δ 都在 0 附近，和 `reseed` 没有区别；最优结果均匀分布在第 0/1/2 轮。
  → 对强编辑器，目前的 loop 只是「多抽几次」。要证明路由有价值，必须加 **best-of-12 纯 reseed** 对照组。

### 发现 3：双向 gate 拦住了大量 reward hacking（创新点 D 有硬证据了）

ip2p 的 1080 个候选中，831 个被 gate 判 0；其中 **559 个在旧的加权和下 > 0.65**（会被当作好结果）。
有 17 个任务，加权和选出的「最佳」比 gate 选出的低 0.05 以上。comp 和 qwen 几乎不触发这个问题（61 和 3 个）。
gate 触发原因：`bg_changed` 26、`bg_semantic` 17、`subject_count` 6、`identity` 6。

## 3. 为什么还要保留 InstructPix2Pix（ip2p）

ip2p 结果一般是预期之内的——它从来不是我们要贡献的编辑器，而是**实验工具**。保留它的理由都是上面的数据：

1. **它是唯一能检验闭环的编辑器。** 强编辑器第 0 轮就饱和（comp 一次生成 0.83，qwen 0.79），loop 没有空间；
   ip2p 的 gate 通过数能从 19 拉到 58，「诊断 → 动作」和跨样本动作记忆（UCB）只有在它身上才能测出差别。
2. **它天然产生 reward hacking 样本。** 559 个「加权和高、实际没换背景」的候选是 D 的核心证据；
   comp/qwen 不产生这类失败，没有 ip2p 就没法做 E3（hacking 分析）。
3. **它提供分数跨度。** 做 E2（与人的一致性）需要从差到好的完整分布；只有 comp 和 qwen 的话分数都挤在 0.8–0.95，
   Spearman 没意义。
4. **它是「指令编辑器」这一类的下界。** E6 想展示的是权衡：ip2p（不换背景）– comp（主体完美、物理差）– qwen（物理好、主体有漂移）。
5. **便宜。** 8 GB 笔记本就能跑，调路由策略时可以在 ip2p 上快速迭代，再上 H200 跑 qwen。

**定位调整**：ip2p 不再出现在「谁的编辑最好」的主表里，只用在 E3（hacking）、E4 的「弱编辑器」一栏、E2 的负样本。
不需要再为它调参或做改进。

## 4. 如果 ip2p 一般，创新点在哪里

创新点从来不在编辑器上，而在**怎样评价背景替换**以及**评价如何驱动修复**。根据现在的数据重新排序：

| | 创新点 | 现在的证据 | 还缺什么 |
|---|---|---|---|
| **A（主线，加强）** | Invariant–covariant critic：保留型 critic 偏爱复制粘贴，协变感知的 critic 能纠正 | 发现 1：comp 赢 75/90 全靠 keep；light_harmony 偏向原图光照 | E2 人工标注确认人偏好 qwen；改写 appearance / light_harmony 后重新打分，看排序是否翻转 |
| **C（已成型）** | 可控物理违例集 + 逐 critic AUROC，包括诚实的负结果（缺接触阴影、轻微下沉谁都看不出） | E1 v2 | 加上通用 reward model（EditReward / VIEScore）做对照 |
| **D（有硬证据）** | 双向 gate + 门控几何平均防 hacking | 发现 3：559 / 831 | held-out 评估器曲线（E3） |
| **E（需要改方向）** | 诊断 → 动作路由 | 发现 2：参数动作有效，prompt 子句无效或有害 | 纯 reseed 对照；**换成局部修复动作**（见下） |
| **F（新）** | **Critic 引导的混合编辑**：comp 保证主体不变，再只在接触带 C / 边界带 B 用 qwen（或 SDXL inpaint）做局部修复 | comp 的 keep ≈ 1、qwen 的 world 更好，两者互补 | 实现 `RegionRepair` 动作并跑 90 任务 |

F 是从这批数据里直接长出来的：comp 的问题（主体悬浮、没有阴影、物体贴着轮廓）都在 C/B 区域，而 S/B/C/G 的区域划分已经有了。
如果 hybrid 能同时拿到 comp 的 keep 和 qwen 的 world，那就是「诊断 → 区域化修复」比「诊断 → 改 prompt」更好的直接证据，
E 也就不再只是 demo。

建议的一句话版本（替换旧版）：

> 背景替换的评价不能只看「主体有没有变」：我们发现保留型 critic 系统性地偏爱复制粘贴，把没有物理耦合的结果排在前面。
> 我们提出从主体 mask 自动生成的 invariant–covariant 区域 critic，用可控违例集和人工标注验证它；
> 并用它驱动区域化修复（粘贴主体 + 只修接触带），在相同编辑预算下比改 prompt 的闭环更好、也更不容易被 hack。

## 5. 下一步（按优先级）

### P0：决定叙事（本周，10/4–10/10）

1. **E2 人工标注（最先做，不需要 GPU）。** 90 个任务的 comp vs qwen 成对偏好 + 每张图 4 个是/否（主体是否被改、是否站在地面上、
   有没有接触阴影、光照是否一致），再加约 30 张 ip2p 作为低分样本。两人标注，算 κ。约 150 张图、每人 2–3 小时。
   → 如果人偏好 qwen，发现 1 成立，A 是主线；如果人偏好 comp，说明 critic 是对的，主线退回 C + D。
2. **修协变 critic，然后 rescore（CHTC，只打分不编辑，很便宜）。**
   * `appearance`：只比结构和纹理（DINOv2 patch / 梯度 / 边缘），不比颜色和明暗；或改成「局部光照归一化」（每个小块减均值）。
   * `light_harmony`：参照物从「原图」改成「新背景」——比较主体亮部色温 / 亮侧方向与新背景 G 区估计出的光源。
   * 姿态改变要继续罚（silhouette 保持不变）。
   * `bash chtc/pack_rescore.sh ip2p_v2 comp_v2 qwen` → `MODE=rescore`。看 comp vs qwen 的排序和 E2 的一致性是否提高。
3. **修 `chtc/collect.sh`**：不要直接解包覆盖 `summary.csv`，解包后从 `report.json` 重建；把 `experiments/analyze_runs.py` 接到 collect 后面。

### P1：证明闭环的价值（10/11–10/24）

4. **等预算对照（E4）**：每个编辑器加两个 run——`oneshot`（1 次）和 `bestof12`（12 个 seed、同一个 prompt、不路由）。
   router 必须打败 `bestof12` 才能说闭环有用。ip2p 再加一个只用参数动作的路由（去掉所有 prompt 子句）。
5. **实现 F：`RegionRepair` 动作**。comp 结果 → 在 C（接触带）和 B（边界带）上做 mask inpaint（qwen-image-edit 带 mask 或 SDXL inpaint，
   prompt 写明 "the subject rests on {support} with a soft contact shadow"）。路由规则：`floating / missing_shadow / implausible` → C 区修复，
   `halo / old_bg_residue` → B 区修复。90 任务 × 12 次编辑，H200。
6. **E1 加通用 reward 对照**：在违例集上跑 EditReward 或 VIEScore（Qwen2.5-VL 版），说明「通用 reward 看不出物理错误」。

### P2：补足弱项（10/25 起）

7. **数据扩充到 ≥ 100 张**（计划里 W2 的任务，目前仍是 15 张）。重点补支撑面类型（草地、沙、路面、室内地板），以及站姿的狗。
8. **阴影和下沉**：两项 E1 负结果。试一个专门的阴影检测模型，或给 VLM 放大 + 增强对比度的接触区裁剪图。
9. **VLM 方差**：每个问题采样 3 次，报告分数的可复现性（审计里提过，还没做）。

### 止损点

* E2 显示人更喜欢 comp → 放弃「critic 偏爱复制粘贴」的叙事，主线改成 C（评测分析）+ D（防 hacking）。
* `bestof12` 打平或打败 router，且 F 也没有提升 → 闭环部分只保留 ip2p 上的参数路由结果，作为次要发现。

## 6. 和 10 周计划的对照

现在是 W1 末。critic、区域、门控、路由、违例集、三个编辑器（原计划 W2–W7）已经提前完成；
**落后的是数据扩充（W2）和人工标注（W8）**，而这两项恰好是现在最卡叙事的。所以接下来两周把它们提前，
再做 E4 对照和 F。
