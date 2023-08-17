# =============================================================================
# Import required libraries
# =============================================================================
import torch
import numpy as np


class EvaluationMetrics():
    def __init__(self, args):
        self.args = args
        self.epsilon = 1e-07

    def per_class_precision(self, targets, outputs):
        tp = torch.sum(targets * outputs, 0)
        predicted = torch.sum(outputs, 0)
        return torch.mean(tp / (predicted + self.epsilon))

    def per_class_recall(self, targets, outputs):
        tp = torch.sum(targets * outputs, 0)
        grand_truth = torch.sum(targets, 0)
        return torch.mean(tp / (grand_truth + self.epsilon))

    def per_image_precision(self, targets, outputs):
        tp = torch.sum(targets * outputs)
        predicted = torch.sum(outputs)
        return tp / (predicted + self.epsilon)

    def per_image_recall(self, targets, outputs):
        tp = torch.sum(targets * outputs)
        grand_truth = torch.sum(targets)
        return tp / (grand_truth + self.epsilon)

    def f1_score(self, precision, recall):
        return 2 * ((precision * recall) / (precision + recall + self.epsilon))

    def N_plus(self, targets, outputs):
        tp = torch.sum(targets * outputs, 0)
        return torch.sum(torch.gt(tp, 0).int())

    def average_precision(self, target, output):
        epsilon = 1e-8

        # sort examples
        indices = output.argsort()[::-1]
        # Computes prec@i
        total_count_ = np.cumsum(np.ones((len(output), 1)))

        target_ = target[indices]
        ind = target_ == 1
        pos_count_ = np.cumsum(ind)
        total = pos_count_[-1]
        pos_count_[np.logical_not(ind)] = 0
        pp = pos_count_ / total_count_
        precision_at_i_ = np.sum(pp)
        precision_at_i = precision_at_i_ / (total + epsilon)
        return precision_at_i

    def mAP(self, targets, outputs):

        targets = np.array(targets.cpu().detach())
        outputs = np.array(outputs.cpu().detach())

        if np.size(outputs) == 0:
            return 0
        ap = np.zeros((outputs.shape[1]))
        # compute average precision for each class
        for k in range(outputs.shape[1]):
            # sort scores
            tar = targets[:, k]
            out = outputs[:, k]
            # compute average precision
            ap[k] = self.average_precision(tar, out)
        return 100 * ap.mean()

    def calculate_metrics(self,
                          targets,
                          outputs,
                          ema_outputs=None,
                          threshold=None):
        # mAP
        m_ap = self.mAP(targets, outputs)
        #
        outputs = torch.gt(outputs, threshold).float()
        pcp = self.per_class_precision(targets, outputs)
        pcr = self.per_class_recall(targets, outputs)
        pcf = self.f1_score(pcp, pcr)
        n_plus = self.N_plus(targets, outputs)
        if ema_outputs is not None:
            ema_m_ap = self.mAP(targets, ema_outputs)
            ema_outputs = torch.gt(ema_outputs, threshold).float()
            ema_pcp = self.per_class_precision(targets, ema_outputs)
            ema_pcr = self.per_class_recall(targets, ema_outputs)
            ema_pcf = self.f1_score(ema_pcp, ema_pcr)
            ema_n_plus = self.N_plus(targets, ema_outputs)
        return {'per_class/precision': pcp,
                'per_class/recall': pcr,
                'per_class/f1': pcf,
                'N+': n_plus,
                'm_ap': m_ap,
                'ema_per_class/precision': ema_pcp if ema_outputs is not None else 0,
                'ema_per_class/recall': ema_pcr if ema_outputs is not None else 0,
                'ema_per_class/f1': ema_pcf if ema_outputs is not None else 0,
                'ema_N+': ema_n_plus if ema_outputs is not None else 0,
                'ema_m_ap': ema_m_ap if ema_outputs is not None else 0
                }
