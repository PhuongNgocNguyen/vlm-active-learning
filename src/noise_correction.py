"""Noise-transition estimation and correction for VLM-generated weak labels."""

import numpy as np
import torch
import torch.nn.functional as F


def estimate_transition_matrix(y_train, selected_indices, num_super_classes):
    """Estimate P(predicted weak label | true weak label) from trusted samples."""
    print(f"Calculating transition matrix from {len(selected_indices)} samples...")

    predicted_labels = y_train['weak_pred'][selected_indices]
    true_labels = y_train['weak_gt'][selected_indices]
    counts = np.zeros((num_super_classes, num_super_classes), dtype=float)

    for predicted_label, true_label in zip(predicted_labels, true_labels):
        if predicted_label == -1 or true_label == -1:
            continue
        counts[true_label, predicted_label] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def forward_corrected_loss(
    logits,
    targets,
    transition_matrix=None,
    ignore_index=-1,
    reduction='mean',
):
    """Apply forward loss correction when a transition matrix is available."""
    if transition_matrix is None:
        loss = F.cross_entropy(
            logits,
            targets,
            ignore_index=ignore_index,
            reduction=reduction,
        )
        return loss, logits

    probabilities = torch.softmax(logits, dim=1)
    corrected_probabilities = torch.matmul(probabilities, transition_matrix)
    log_corrected = torch.log(corrected_probabilities + 1e-8)
    loss = F.nll_loss(
        log_corrected,
        targets,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    return loss, corrected_probabilities
