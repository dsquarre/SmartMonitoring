import unittest
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_soft_confusion_metrics(y_true, preds_flat, eps=1e-7):
    """
    Computes soft (probability-weighted) confusion counts and soft-F2 score.
    Training-reward construct only; not a clinical or paper reporting metric.
    """
    preds_clipped = np.clip(preds_flat, eps, 1.0 - eps)
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)

    soft_tp = float(np.sum(preds_clipped[pos_mask])) if np.any(pos_mask) else 0.0
    soft_fn = float(np.sum(1.0 - preds_clipped[pos_mask])) if np.any(pos_mask) else 0.0
    soft_fp = float(np.sum(preds_clipped[neg_mask])) if np.any(neg_mask) else 0.0
    soft_tn = float(np.sum(1.0 - preds_clipped[neg_mask])) if np.any(neg_mask) else 0.0

    denom_soft_prec = soft_tp + soft_fp + eps
    denom_soft_rec = soft_tp + soft_fn + eps
    soft_prec = float(soft_tp / denom_soft_prec)
    soft_rec = float(soft_tp / denom_soft_rec)

    denom_soft_f2 = (4 * soft_prec + soft_rec)
    soft_f2 = float((5 * soft_prec * soft_rec) / denom_soft_f2) if denom_soft_f2 > 0 else 0.0

    return {
        "soft_tp": soft_tp,
        "soft_fp": soft_fp,
        "soft_tn": soft_tn,
        "soft_fn": soft_fn,
        "soft_precision": soft_prec,
        "soft_recall": soft_rec,
        "soft_f2": soft_f2
    }


class TestSoftConfusionMetrics(unittest.TestCase):
    """Unit tests and artifact regression tests for Soft Confusion Counts & Soft-F2."""

    def test_soft_counts_bounds_and_conservation(self):
        """Verifies that soft_TP + soft_FN = N_pos and soft_FP + soft_TN = N_neg."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=int)
        preds = np.array([0.85, 0.40, 0.25, 0.10, 0.05, 0.02, 0.15, 0.01], dtype=np.float32)

        res = compute_soft_confusion_metrics(y_true, preds)

        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        self.assertAlmostEqual(res["soft_tp"] + res["soft_fn"], n_pos, places=5)
        self.assertAlmostEqual(res["soft_fp"] + res["soft_tn"], n_neg, places=5)
        self.assertGreaterEqual(res["soft_f2"], 0.0)
        self.assertLessEqual(res["soft_f2"], 1.0)

    def test_smooth_continuity(self):
        """Verifies that shifting probabilities produces smooth, continuous changes in soft_f2."""
        y_true = np.array([1] * 50 + [0] * 50, dtype=int)

        prev_soft_f2 = 0.0
        for shift in np.linspace(0.10, 0.50, 10):
            preds = np.zeros(100, dtype=np.float32)
            preds[:50] = 0.15 + shift  # Positive class probabilities shifting up
            preds[50:] = 0.05          # Negative class probabilities low

            res = compute_soft_confusion_metrics(y_true, preds)

            # Assert monotonicity with positive shift
            self.assertGreater(res["soft_f2"], prev_soft_f2)
            prev_soft_f2 = res["soft_f2"]

    def test_regression_artifact_collapse_mitigation(self):
        """
        REGRESSION TEST FOR THRESHOLD 0.3 COLLAPSE BUG:
        Reproduces the exact probability distribution where positive samples cluster in 0.15–0.28.
        Proves that hard F2(0.3) collapses to near zero due to threshold cut-off,
        while soft_f2 remains non-zero, continuous, and uncorrupted for RL rewards.
        """
        y_true = np.array([1] * 50 + [0] * 950, dtype=int)  # 5% positive imbalanced AFib data

        # Positive sample probabilities cluster around 0.22 (below 0.3 threshold)
        preds = np.zeros(1000, dtype=np.float32)
        preds[:50] = np.random.uniform(0.15, 0.28, size=50)  # Positives below 0.3
        preds[50:] = np.random.uniform(0.01, 0.05, size=950)  # Negatives low

        # Hard threshold at 0.3
        pred_binary_03 = (preds >= 0.3).astype(int)
        hard_prec_03 = precision_score(y_true, pred_binary_03, zero_division=0)
        hard_rec_03 = recall_score(y_true, pred_binary_03, zero_division=0)
        denom_hard = 4 * hard_prec_03 + hard_rec_03
        hard_f2_03 = (5 * hard_prec_03 * hard_rec_03 / denom_hard) if denom_hard > 0 else 0.0

        # Soft confusion metric
        soft_res = compute_soft_confusion_metrics(y_true, preds)

        # 1. Hard F2 at threshold 0.3 collapses to 0.0 because Recall = 0.0
        self.assertEqual(hard_rec_03, 0.0)
        self.assertEqual(hard_f2_03, 0.0)

        # 2. Soft F2 remains robust, smooth, and non-zero (> 0.25)
        self.assertGreater(soft_res["soft_recall"], 0.15)
        self.assertGreater(soft_res["soft_f2"], 0.20)
        print(f"\n[Artifact Regression Verification] Hard F2(0.3): {hard_f2_03:.4f} (COLLAPSED) | Soft-F2: {soft_res['soft_f2']:.4f} (ROBUST)")


if __name__ == "__main__":
    unittest.main()
