# 通用 reward model 基线、VLM 稳定性与第二个 VLM：协议与运行方法

*日期：2026-11-05。全部只打分、不编辑。**截至本文写下，这些脚本只在 CPU 上用假打分器和假数据测过（18 个单元测试），
没有在 GPU 上跑过任何一个模型**；下面标“未验证”的地方要靠第一次 CHTC 冒烟来确认。*

## 0. 要回答什么，以及事先定好的判定

| 问题 | 指标 | 判定（先定，再看结果） |
|---|---|---|
| Q1 通用 reward model 能分出物理违例吗？ | 违例集（15 张源图 × 7 类违例）上，干净粘贴（null；halo 用 foreign_clean）vs 违例的 AUROC 和同图配对准确率，按源图做 cluster bootstrap 95% CI | 某个基线在某类违例上 AUROC 的 CI 下界 > 0.5，才算“能看出”；CI 跨 0.5 就不能说看得出 |
| Q2 我们的 critic 是否看到了基线看不到的？ | 同一违例类别上，我们的 critic（`runs/e1v2/auroc.csv`）与基线并列 | 只有“基线 CI 跨 0.5 且我们的明显 > 0.5”才写成“基线看不出、我们看得出”；两边都 ≈ 0.5 就写“都看不出”，不写优势；基线更高就如实写，并把卖点改成“更便宜 / 可解释” |
| Q3 VLM critic 的答案稳不稳？ | 每个问题在 6 种视角下的标准差和极差；`orig_rep`（同一输入重复）给非确定性下限 | **建议阈值（事先设定，可改）**：某问题在 comp vs qwen 比较上，同一任务的胜负符号在所有视角下都不变的任务比例 < 80%，则该 VLM critic 不用于排序结论 |
| Q4 结论依赖于哪个 VLM？ | 第二个 VLM（`vlm2`）重答同样的问题，比较 comp vs qwen 的胜负方向、与第一个 VLM 的一致性 | 两个 VLM 对同一问题的胜负方向相反，则该问题不用于主张 |

## 1. 重要的设计决定

- **“重复采样 3 次”没有意义**。VLM critic 读的是下一个 token 的 logits，同一输入每次给同一个数。
  可变的是输入：问题措辞、裁剪框的外扩、图像分辨率。所以做的是**视角扰动**（原措辞、两种改写、更紧 / 更松的裁剪、
  降分辨率），外加一次原样重复（`orig_rep`）来量非确定性下限。
- **VIEScore 是“风格”复现**：用本仓库自己写的提示词，让 VLM 打 0–10 分（编辑成功度、过度编辑、自然度、伪影），
  SC、PQ 取各自的最小值，overall = sqrt(SC×PQ)，这是 VIEScore 论文描述的合成方式。**不是官方提示词**，
  论文里要写成 “VIEScore-style”，并写明用的是哪个 VLM。
- **EditReward 用官方接口**（`EditRewardInferencer`，`reward_dim="overall_detail"`，`rm_head_type="ranknet_multi_head"`，
  权重 `TIGER-Lab/EditReward-Qwen2.5-VL-7B`，MIT 许可）。它要求 `transformers==4.57.0`、`trl==0.8.6`，与本仓库
  （`transformers>=5.5`）冲突，所以 `MODE=editreward` 在**单独的 venv** 里跑。README 没有说明分数范围，
  只比排序 / AUROC，不比绝对值。
- **第二个 VLM**：默认 `OpenGVLab/InternVL3-8B-hf`（无需 token）。注意它的语言模型也是 Qwen2.5 系，
  与 Qwen2.5-VL 只是**部分独立**。想要完全不同的家族可用 `--set models.vlm2.id=google/gemma-3-12b-it`
  （需要 HF_TOKEN 并接受许可；该 id 我没有核实）。
- 违例集的 instruction 是通用的（“把主体放进场景，使结果看起来像自然、物理合理的照片”），因为违例图并没有换背景；
  这对依赖 instruction 的 reward model 有影响，写论文时要说明。
- 违例集正样本 null 与负样本走同一条抠图-补背景-贴回流水线，所以基线分开它们不是靠“有没有粘贴痕迹”。

## 2. 文件

| 文件 | 作用 |
|---|---|
| `experiments/build_scoring_set.py` | 从 E1 的 `examples/` 或编辑器 run 构建打分集（manifest + images），CPU 即可 |
| `experiments/score_viescore.py` | VIEScore 风格 judge（`--vlm-kind vlm|vlm2`） |
| `experiments/score_editreward.py` | EditReward（独立环境） |
| `experiments/reward_report.py` | 合并分数，输出 AUROC + CI、与我们 critic 并列的表、编辑器 run 的胜负与 Spearman |
| `experiments/vlm_variance.py` | VLM 稳定性（视角扰动）与 comp/qwen 胜负是否稳定 |
| `src/baselines/` | 上述逻辑（提示词、解析、统计、报告） |
| `chtc/run.sh` 新模式 `viescore` / `editreward` / `vlmvar`；`chtc/pack_rewardset.sh` | CHTC 入口与打包 |

## 3. 运行步骤

### 3.1 本机（CPU，已建好）

```bash
python experiments/build_scoring_set.py --name rw_violations --violations runs/e1v2      # 135 项 = 15 张图 × 9 种变体
python experiments/build_scoring_set.py --name rw_runs --runs runs/comp_v2 runs/qwen runs/ip2p_v2   # 270 项
bash chtc/pack_rewardset.sh rw_violations                                                  # -> chtc/rewardset_in.tar.gz
```

构建 run 集时，`runs/qwen/adult_stand_mountain_01__to_snow/before.png` 本地文件被截断（1,191,936 字节，
其余 run 里同一张是 1,650,238 字节且 md5 一致），脚本自动用 `comp_v2` 的同一张代替并打印警告。
**这份损坏的原件没有被改动**；如果你要重新跑 rescore，建议从 CHTC 上的结果包重新解压它。

### 3.2 CHTC

先 `git push` 本机提交，在提交节点 `git pull && bash chtc/pack.sh`，再把打包好的数据集从本机传上去
（`<netid>` 和提交节点上仓库的位置按你的实际情况改）：

```bash
scp chtc/rewardset_in.tar.gz <netid>@ap2001.chtc.wisc.edu:~/self_improving_editor/chtc/
```

```bash
# 冒烟：先各跑 3 项确认环境能起来（EXTRA 传给脚本）
condor_submit chtc/job.sub MODE=viescore   RUN=rw_vio_vie_smoke LIST=jobs/one.txt EXTRA_IN=,chtc/rewardset_in.tar.gz GPUMEM=24000M EXTRA="--limit 3"
condor_submit chtc/job.sub MODE=editreward RUN=rw_vio_er_smoke  LIST=jobs/one.txt EXTRA_IN=,chtc/rewardset_in.tar.gz GPUMEM=24000M EXTRA="--limit 3"
# 正式
condor_submit chtc/job.sub MODE=viescore   RUN=rw_vio_vie  LIST=jobs/one.txt EXTRA_IN=,chtc/rewardset_in.tar.gz GPUMEM=24000M
condor_submit chtc/job.sub MODE=viescore   RUN=rw_vio_vie2 LIST=jobs/one.txt EXTRA_IN=,chtc/rewardset_in.tar.gz GPUMEM=24000M VLM_KIND=vlm2
condor_submit chtc/job.sub MODE=editreward RUN=rw_vio_er   LIST=jobs/one.txt EXTRA_IN=,chtc/rewardset_in.tar.gz GPUMEM=24000M
# VLM 稳定性（用已有的 rescore 包）
bash chtc/pack_rescore.sh comp_v2 qwen
condor_submit chtc/job.sub MODE=vlmvar RUN=vlmvar  LIST=jobs/one.txt EXTRA_IN=,chtc/rescore_in.tar.gz EXTRA="--limit 3"   # 冒烟
condor_submit chtc/job.sub MODE=vlmvar RUN=vlmvar  LIST=jobs/one.txt EXTRA_IN=,chtc/rescore_in.tar.gz
condor_submit chtc/job.sub MODE=vlmvar RUN=vlmvar2 LIST=jobs/one.txt EXTRA_IN=,chtc/rescore_in.tar.gz VLM_KIND=vlm2
```

结果回到提交节点的 `results/result_<RUN>_dog_sit_river_01.tar.gz`。每个 job 结束后从本机取回（把 `<RUN>` 换成对应名字）：

```bash
scp <netid>@ap2001.chtc.wisc.edu:~/self_improving_editor/results/result_<RUN>_dog_sit_river_01.tar.gz results/
tar xzf results/result_<RUN>_dog_sit_river_01.tar.gz        # -> runs/<RUN>/
```

然后在本机：

```bash
python experiments/reward_report.py --scoreset runs/rw_violations --name rw_violations_report \
    --scores runs/rw_vio_vie/scores_viescore_vlm.jsonl runs/rw_vio_vie2/scores_viescore_vlm2.jsonl runs/rw_vio_er/scores_editreward.jsonl \
    --critics-auroc runs/e1v2/auroc.csv
python experiments/vlm_variance.py --report-only runs/vlmvar/variance.csv --name vlmvar_report
```

## 4. 未验证 / 已知风险

1. **EditReward 的环境从未在 CHTC 上建成过。** `pip install --no-deps -e` 之后缺哪些依赖、容器里有没有 `git`
   （没有则退回用 python 下载源码包）、`trl==0.8.6` 与 `transformers==4.57.0` 能否共存，都要看冒烟日志。
   `reward[0]` 的含义取自 README 示例；单样本调用的返回形状我没有核实。
2. **InternVL3 经 `VLMScorer`（`apply_chat_template` + `processor(text=, images=)`）是否能得到正确的 Yes/No logits 没有验证**；
   冒烟里先看 `variance.csv` 的 p_yes 是否在 0–1 之间且不是常数。
3. VIEScore 风格 judge 的回复格式依赖提示词；解析失败时分数记为 None（不记 0），`raw_sc` / `raw_pq` 保留原文，
   看冒烟输出里 None 的比例。
4. 违例集只有 15 张源图，CI 会很宽；N=15 个聚类下“CI 下界 > 0.5”是个严格的条件，很可能出现“看不出但也无法证明看不见”。
5. 我们 critic 的 E1 AUROC（`runs/e1v2/auroc.csv`）是跨样本 AUROC，没有 CI，与基线表里带 CI 的数不完全可比；
   而且其中一些 1.00（如 `appearance` 对 float / scale / sink）来自主体位置改变本身，而不是物理判断。
   并列比较前，最好用同一套 bootstrap 重算我们 critic 的 CI（`raw_scores.json` 里有逐样本分数）。
