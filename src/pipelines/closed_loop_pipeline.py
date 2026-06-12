"""Closed-loop image editing pipeline."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from src.critics.base import Critic
from src.detectors.base import Detector
from src.editors.base import Editor
from src.prompts.refinement import refine_prompt
from src.utils.io_utils import make_experiment_id, save_experiment, save_iteration
from src.utils.visualization import draw_bbox, save_comparison

logger = logging.getLogger("closed_loop_editor")


class ClosedLoopPipeline:
    """
    Orchestrates iterative image editing with evaluation and refinement.

    Flow: Generate → Evaluate → Refine Prompt → Generate Again
    """

    def __init__(
        self,
        detector: Detector,
        editor: Editor,
        critic: Critic,
        output_dir: Path = Path("data/experiments"),
        config: Optional[Dict[str, Any]] = None,
    ):
        self.detector = detector
        self.editor = editor
        self.critic = critic
        self.output_dir = Path(output_dir)
        self.config = config or {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        original_image: np.ndarray,
        base_prompt: str,
        max_iterations: Optional[int] = None,
        score_threshold: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        save_results: bool = True,
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run iterative closed-loop editing.

        Args:
            original_image: Original BGR image.
            base_prompt: Base editing prompt.
            max_iterations: Max refinement loops (from config if None).
            score_threshold: Stop when composite score exceeds this (from config if None).
            metadata: Extra fields stored in result JSON.
            save_results: Persist JSON, images, and visualizations.
            experiment_id: Optional directory name prefix for this run.

        Returns:
            Result dict with best_image (BGR array), best_score, all_iterations, metadata.
        """
        pipeline_cfg = self.config.get("pipeline", {})
        output_cfg = self.config.get("output", {})

        max_iterations = max_iterations or pipeline_cfg.get("max_iterations", 3)
        score_threshold = score_threshold or pipeline_cfg.get("score_threshold", 0.7)
        save_images = output_cfg.get("save_images", True)
        save_visualizations = output_cfg.get("save_visualizations", True)

        exp_id = experiment_id or make_experiment_id(
            job_id=(metadata or {}).get("job_id")
        )
        exp_dir = self.output_dir / exp_id if save_results else None

        if save_results and exp_dir is not None:
            (exp_dir / "images").mkdir(parents=True, exist_ok=True)
            (exp_dir / "iterations").mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting pipeline exp=%s max_iter=%d threshold=%.3f",
            exp_id,
            max_iterations,
            score_threshold,
        )

        best_image: Optional[np.ndarray] = None
        best_score = -1.0
        all_iterations = []
        current_prompt = base_prompt

        for iteration in range(max_iterations):
            logger.info("=== Iteration %d/%d ===", iteration + 1, max_iterations)

            try:
                edited_image = self.editor.edit(original_image, current_prompt)
                score = self.critic.score(original_image, edited_image, current_prompt)

                iteration_data: Dict[str, Any] = {
                    "iteration": iteration,
                    "prompt": current_prompt,
                    "score": float(score),
                    "timestamp": datetime.now().isoformat(),
                }

                if hasattr(self.critic, "get_individual_scores"):
                    individual_scores = self.critic.get_individual_scores(
                        original_image, edited_image, current_prompt
                    )
                    iteration_data["individual_scores"] = individual_scores
                    logger.info("Scores: composite=%.4f detail=%s", score, individual_scores)
                else:
                    logger.info("Composite score: %.4f", score)

                orig_cls, orig_box = self.detector.detect(original_image)
                edit_cls, edit_box = self.detector.detect(edited_image)
                iteration_data["detections"] = {
                    "orig_cls": orig_cls,
                    "orig_box": orig_box,
                    "edit_cls": edit_cls,
                    "edit_box": edit_box,
                }

                if save_results and exp_dir is not None:
                    if save_images:
                        cv2.imwrite(
                            str(exp_dir / "images" / f"iter_{iteration:03d}.jpg"),
                            edited_image,
                        )

                    if save_visualizations:
                        orig_label = (
                            self.detector.get_class_name(orig_cls)
                            if orig_cls is not None
                            else "original"
                        )
                        edit_label = (
                            self.detector.get_class_name(edit_cls)
                            if edit_cls is not None
                            else "edited"
                        )
                        orig_vis = (
                            draw_bbox(original_image.copy(), orig_box, orig_label)
                            if orig_box
                            else original_image
                        )
                        edit_vis = (
                            draw_bbox(edited_image.copy(), edit_box, edit_label)
                            if edit_box
                            else edited_image
                        )
                        save_comparison(
                            orig_vis,
                            edit_vis,
                            output_path=exp_dir / "images" / f"compare_{iteration:03d}.jpg",
                        )

                    save_iteration(iteration_data, exp_dir, iteration)

                all_iterations.append(iteration_data)

                if score > best_score:
                    best_score = score
                    best_image = edited_image
                    logger.info("New best score: %.4f", best_score)

                if score >= score_threshold:
                    logger.info("Reached threshold %.3f, stopping early", score_threshold)
                    break

                scores_for_refine = iteration_data.get(
                    "individual_scores", {"composite": score}
                )
                current_prompt = refine_prompt(
                    base_prompt,
                    scores_for_refine,
                    iteration,
                    max_iterations,
                )
                logger.info("Refined prompt: %s", current_prompt)

            except Exception as exc:
                logger.error("Error in iteration %d: %s", iteration, exc, exc_info=True)
                iteration_data = {"iteration": iteration, "error": str(exc)}
                all_iterations.append(iteration_data)
                if save_results and exp_dir is not None:
                    save_iteration(iteration_data, exp_dir, iteration)

        result: Dict[str, Any] = {
            "experiment_id": exp_id,
            "best_image": best_image,
            "best_score": best_score,
            "all_iterations": all_iterations,
            "metadata": metadata or {},
            "config": {
                "max_iterations": max_iterations,
                "score_threshold": score_threshold,
            },
        }

        if save_results and exp_dir is not None:
            self._save_results(result, exp_dir, best_image if save_images else None)

        return result

    def _save_results(
        self,
        result: Dict[str, Any],
        exp_dir: Path,
        best_image: Optional[np.ndarray],
    ) -> None:
        """Persist final artifacts and metadata JSON."""
        result_copy = {
            "experiment_id": result["experiment_id"],
            "best_score": result["best_score"],
            "all_iterations": result["all_iterations"],
            "metadata": {
                **result["metadata"],
                "timestamp": datetime.now().isoformat(),
                "num_iterations": len(result["all_iterations"]),
            },
            "config": result["config"],
        }

        if best_image is not None:
            final_path = exp_dir / "images" / "final_edit.jpg"
            cv2.imwrite(str(final_path), best_image)
            result_copy["best_image_path"] = str(final_path)
            logger.info("Saved best edit to %s", final_path)

        save_experiment(result_copy, exp_dir)
        logger.info("Saved experiment metadata to %s", exp_dir / "metadata.json")
