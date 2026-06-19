# Grounding DINO MVP — research/mvp-grounding-dino

Minimal pipeline to evaluate whether important objects are preserved after image editing.

## Quick start

```bash
pip install -r requirements.txt
python run_experiment.py --use-mock          # offline CPU test
python run_experiment.py                     # real Grounding DINO (GPU recommended)
python run_experiment.py --sample sample_001 # single sample
```

## Dataset layout

```
data/
  sample_001/
    before.jpg
    after.jpg
    instruction.txt
```

## Output layout

```
results/
  sample_001/
    detections_before.json
    detections_after.json
    scores.json
    refined_prompt.txt
    summary.png
```

## Scoring

Configured in `config/scoring.yaml`:

```
final_score = 0.7 * detection_score + 0.3 * count_score
```

## CHTC

```bash
mkdir -p logs
condor_submit submit.sub
```

## Scope (MVP only)

Implemented: Grounding DINO, Detection Critic, Count Critic, rule-based prompt refinement.

Not implemented on this branch: SAM2, CLIP, VLM, Scene Graphs, Physics Critics, World Models.
