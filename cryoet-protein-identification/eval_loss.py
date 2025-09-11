# -*- coding: utf-8 -*-
import cc3d
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from constants import particle_radius, particle_weights
from scipy.spatial import KDTree


def f_score(pr, gt, beta=4, eps=1e-7, threshold=None, activation="sigmoid"):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    # if activation is None or activation == "none":
    #     activation_fn = lambda x: x
    if activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta**2) * tp + eps) / (
        (1 + beta**2) * tp + beta**2 * fn + fp + eps
    )

    return score


class DiceLoss(nn.Module):
    __name__ = "dice_loss"

    def __init__(self, eps=1e-7, activation="sigmoid"):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(
            y_pr,
            y_gt,
            beta=4.0,
            eps=self.eps,
            threshold=None,
            activation=self.activation,
        )


class BCEDiceLoss(DiceLoss):
    __name__ = "bce_dice_loss"

    def __init__(
        self,
        device,
        eps=1e-7,
        activation="sigmoid",
        lambda_dice=1.0,
        lambda_bce=1.0,
    ):
        super().__init__(eps, activation)
        if activation is None:
            self.bce = nn.BCELoss(
                reduction="mean",
                pos_weight=torch.tensor([1, 1, 1, 1, 1])
                .reshape((5, 1, 1, 1))
                .to(device),
            )
        else:
            self.bce = nn.BCEWithLogitsLoss(
                reduction="mean",
                pos_weight=torch.tensor([1, 1, 1, 1, 1])
                .reshape((5, 1, 1, 1))
                .to(device),
            )
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt[:, 1:, :, :, :])
        bce = self.bce(y_pr, y_gt[:, 1:, :, :, :])
        return (self.lambda_dice * dice) + (self.lambda_bce * bce)


def np_find_component(probability, threshold):
    num_particle_type, D, H, W = probability.shape
    binary = probability > np.array(threshold).reshape(
        num_particle_type, 1, 1, 1
    )
    component = np.zeros((num_particle_type, D, H, W), np.uint32)
    for i in range(num_particle_type):
        component[i] = cc3d.connected_components(binary[i], connectivity=6)
    return component


def np_find_centroid(component):
    centroid = []
    num_particle_type, D, H, W = component.shape
    for i in range(num_particle_type):
        stats = cc3d.statistics(component[i])
        zyx = stats["centroids"][1:]
        xyz = np.ascontiguousarray(zyx[:, ::-1])
        centroid.append(xyz * 10.012444)
    return centroid


class ParticipantVisibleError(Exception):
    pass


def compute_metrics(reference_points, reference_radius, candidate_points):
    num_reference_particles = len(reference_points)
    num_candidate_particles = len(candidate_points)

    if len(reference_points) == 0:
        return 0, num_candidate_particles, 0

    if len(candidate_points) == 0:
        return 0, 0, num_reference_particles

    ref_tree = KDTree(reference_points)
    candidate_tree = KDTree(candidate_points)
    raw_matches = candidate_tree.query_ball_tree(ref_tree, r=reference_radius)
    matches_within_threshold = []
    for match in raw_matches:
        matches_within_threshold.extend(match)
    # Prevent submitting multiple matches per particle.
    # This won't be be strictly correct in the
    # (extremely rare) case where true particles
    # are very close to each other.
    matches_within_threshold = set(matches_within_threshold)
    tp = int(len(matches_within_threshold))
    fp = int(num_candidate_particles - tp)
    fn = int(num_reference_particles - tp)
    return tp, fp, fn


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    distance_multiplier: float,
    beta: int,
) -> float:
    """
    F_beta
      - a true positive occurs when
         - (a) the predicted location is within a threshold
         of the particle radius
         - (b) the correct `particle_type` is specified
      - raw results (TP, FP, FN) are aggregated
      across all experiments
      for each particle type
      - f_beta is calculated for each particle type
      - individual f_beta scores are weighted by particle type for final score
    """

    particle_radius_multiplied = {
        k: v * distance_multiplier for k, v in particle_radius.items()
    }

    # Filter submission to only contain experiments found in the solution split
    split_experiments = set(solution["experiment"].unique())
    submission = submission.loc[
        submission["experiment"].isin(split_experiments)
    ]

    # Only allow known particle types
    if not set(submission["particle_type"].unique()).issubset(
        set(particle_weights.keys())
    ):
        raise ParticipantVisibleError("Unrecognized `particle_type`.")

    assert solution.duplicated(subset=["experiment", "x", "y", "z"]).sum() == 0
    assert particle_radius_multiplied.keys() == particle_weights.keys()

    results = {}
    for particle_type in solution["particle_type"].unique():
        results[particle_type] = {
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 0,
        }

    for experiment in split_experiments:
        for particle_type in solution["particle_type"].unique():
            reference_radius = particle_radius_multiplied[particle_type]
            select = (solution["experiment"] == experiment) & (
                solution["particle_type"] == particle_type
            )
            reference_points = solution.loc[select, ["x", "y", "z"]].values

            select = (submission["experiment"] == experiment) & (
                submission["particle_type"] == particle_type
            )
            candidate_points = submission.loc[select, ["x", "y", "z"]].values

            if len(reference_points) == 0:
                reference_points = np.array([])
                reference_radius = 1

            if len(candidate_points) == 0:
                candidate_points = np.array([])

            tp, fp, fn = compute_metrics(
                reference_points, reference_radius, candidate_points
            )

            results[particle_type]["total_tp"] += tp
            results[particle_type]["total_fp"] += fp
            results[particle_type]["total_fn"] += fn

    aggregate_fbeta = 0.0
    partial_fbeta = {}
    for particle_type, totals in results.items():
        tp = totals["total_tp"]
        fp = totals["total_fp"]
        fn = totals["total_fn"]

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        fbeta = (
            (1 + beta**2)
            * (precision * recall)
            / (beta**2 * precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        aggregate_fbeta += fbeta * particle_weights.get(particle_type, 1.0)
        partial_fbeta[particle_type] = fbeta

    if particle_weights:
        aggregate_fbeta = aggregate_fbeta / sum(particle_weights.values())
    else:
        aggregate_fbeta = aggregate_fbeta / len(results)
    return aggregate_fbeta, partial_fbeta
