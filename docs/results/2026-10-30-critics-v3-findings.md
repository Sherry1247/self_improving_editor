# Critics v3：构造性测试、真实结果与修订

*日期：2026-10-30。数据：CHTC 上 `rescored_v3c`、`rescored_v3d`（对 comp_v2 和 qwen 各 90 个任务重新打分）、`covtest_v2c/v3c/v3d`
（pasted 与 re-lit 对照）。前置：[full-budget v2 结果与计划](./2026-10-03-full-budget-v2-runs-and-plan.md)。*

## 1. 结论

1. **v3 没有改变排序。** comp 赢 qwen：v2 75/90 → v3c 78/90 → v3d 76/90。keep 分支 comp 赢 90/90（0.985 对 0.789）；
   follow 分支 qwen 赢 77/90（0.987 对 0.953）；world 分支 comp 赢 71/90（0.786 对 0.634）。
2. **“critic 偏爱复制粘贴”不能再作为主张。** 看图（`runs/_analysis/review_v3/`）：qwen 在多数 comp 领先最多的任务里确实改了主体
   （外套/背包细节、狗的姿态，appearance 0.00）；comp 的主体像素不变，但光照与新背景不协调（暖色狗在冷蓝室内、暗青色人物在雪地）。
   两种编辑器各有一类真实问题，critic 的分数大体反映了这一点。更稳妥的论点是 **主体保持与物理耦合之间存在权衡**。
3. **v3 的 `light_harmony` 差距是测量不对等造成的，不是光照差异。** 协变项只在“场景光照变化够大”时可检验：
   qwen 83 % 的任务可检验，comp 只有 18 %。可检验时两边得分完全一样（0.408 对 0.407）；
   不可检验的任务回落到 v2 的失配分，得到近乎满分（comp 0.979）。所以整体差距（0.877 对 0.482）来自可检验比例。
4. **协变项与独立 VLM critic 不一致。** qwen 的 `follow` ≈ 0（-0.002，comp 0.053），但 `vlm_lighting` qwen 0.926 对 comp 0.874（qwen 赢 40/16），
   `vlm_surface` 0.961 对 0.915。因此协变项不能作为默认评分的一部分。
5. **appearance v3 在 qwen 上仍很低（0.548）。** 对齐不是主因（对齐后 zncc 仅 0.556 → 0.595，对齐不可靠的任务 0 %）；
   像素级高频相关对生成式编辑器的重新编码很敏感，而 DINOv2 的 `texture`（0.912）与 `identity`（0.992）认为主体基本一致。

## 2. 构造性测试（不依赖人工标注）

保守版重新打光（Lab 空间的亮度/侧光/平均色度，qwen 的 dev 60 + test 30）：

| | relit 更高 / 更低 | 均值 pasted → relit |
|---|---|---|
| appearance | 9 / 16（dev），34 平 | 0.592 → 0.592（不再惩罚重新打光） |
| light_harmony（v3 版） | 38 / 10（dev）、18 / 6（test） | 0.499 → 0.645、0.448 → 0.638 |
| world | 36 / 19（dev）、16 / 12（test） | 0.629 → 0.671、0.637 → 0.705 |
| vlm_lighting / integration / shadow | 略偏向 pasted（降 1–2 点） | 独立 VLM 并不认为重新打光更好 |

注意：light_harmony v3 是为奖励重新打光而设计的，通过这个测试相当于通过单元测试，不是独立证据；
独立的 VLM critic 未见改善，说明测试操作本身可能仍不自然，或 VLM 对这种光照变化不敏感。

## 3. 代码修订（本次）

| 改动 | 原因 |
|---|---|
| `light_harmony` 默认回到 v2（失配分，E1 上颜色偏移 AUROC 1.00）；v3 仅作消融 | 结论 3 |
| 新增探索性 `light_cochange`：不可检验时返回 n/a，不再给“免费的 1.0” | 结论 3 |
| `aggregation.exclude: [light_cochange, unchanged_bg]`：探索性 critic 只记录，不参与任何分数 | 结论 4 |
| appearance：多尺度 ZNCC（默认 scales=(1,2)），并记录每个尺度的值（`zncc_by_scale`）供离线选择 | 结论 5 |
| covtest 新增 `degraded` 对照（轻微模糊 + 亚像素位移 + JPEG，身份不变）：appearance 应基本不变 | 用“身份不变的干扰”而不是“哪个编辑器赢”来标定容忍度 |
| `rescore.py --with`：额外记录探索性 critic 的分数与证据列 | 便于分析 |

## 4. 约束（避免循环论证）

* 阈值、尺度只能在 dev（`_01`、`_02`）上用 `degraded` 对照标定，`_03`（test）只报告。
* 不能为了让 qwen 排在前面而调 critic。标定依据只能是构造性对照（null、degraded、relit）。
* 在没有人工偏好之前，不对“哪个编辑器更好”下结论。

## 5. 修订后的结果（`rescored_v3e`、`covtest_v3e`，代码 4bdd82d）

**噪声下限（NULL：同样的 Lab 往返、强度 0）**：overall 最大 0.020，world 最大 0.057；VLM critic 均值抖动 0.004–0.013，个别任务最大 0.12
（`vlm_lighting` 0.124、`vlm_integration` 0.117）。所以 VLM critic 在 pasted 与 relit 之间 1–2 个百分点的差异在噪声范围内，不能当证据。

**容忍度（degraded：模糊 + 亚像素位移 + JPEG，身份不变）**：comp 的 appearance 1.000 → 0.915（ZNCC 尺度 1/2/4：0.914/0.952/0.979），
texture、identity 基本不变；qwen 的 appearance 0.557 → 0.550。**因此上一节结论 5 需要修正**：qwen 的 appearance 偏低不是被重新编码的像素噪声造成的——
qwen 在粗尺度上的结构相关也只有 0.62（尺度 4），对退化几乎不敏感。更合理的解读是 qwen 确实改变了主体的结构（姿态、细节），与看图一致。
appearance v3 的容忍度（scales=(1,2)）不需要再放宽。

**relit 与 pasted（默认 critic）**：appearance 不惩罚重新打光（qwen 0.602 → 0.602）；默认的 `light_harmony`（v2 失配分）轻微偏向 relit（+0.02）；
`world` 持平或略降（comp dev 0.819 → 0.796，qwen dev 0.802 → 0.784），因此**不能说 relit 更好**。探索性的 `light_cochange` 在可检验时强烈偏向 relit
（comp 0.22 → 0.60），但可检验比例 comp 17–20 %、qwen 83 %。

**排序（light_harmony 回到 v2 后，90 任务）**：

| | comp | qwen | qwen 赢 |
|---|---|---|---|
| 总分 | 0.914 | 0.849 | 22/90（v3d 是 14/90，v2 是 15/90） |
| keep | 0.985 | 0.792 | 0/90 |
| follow | 0.953 | 0.987 | 77/90 |
| world | 0.820 | 0.809 | 41/90（dev 0.822 对 0.803，test 0.816 对 0.821） |
| follow × world（去掉 keep） | 0.882 | 0.890 | 48/90 |

comp 的领先**完全来自 keep**；去掉 keep 后两者打平，world 基本持平。这支持“主体保持与物理耦合是一对权衡”，不是“critic 偏爱复制粘贴”。

**旧场景残留（`unchanged_bg`，探索性）**：comp 反而更多——背景里与原图几乎相同的大块区域占比均值 comp 0.106 对 qwen 0.028，超过 10 % 的任务 33 对 9。
“qwen 保留栏杆”是个别现象，不是普遍现象；且 comp 的高值需要目视确认（可能是合理保留的背景，也可能是 inpainting 没改），先不进评分。
下一步：`MODE=residue` 生成叠加图。
