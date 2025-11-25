"""
These are utilitiy functions for the  2-SAT solver process.
This includes steps for evaluation, DDI computation, and post-processing
our predictions.
"""

from sklearn.metrics import (
    jaccard_score,
    roc_auc_score,
    precision_score,
    f1_score,
    average_precision_score,
)
import numpy as np
import sys
import warnings
import dill

warnings.filterwarnings("ignore")


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def multi_label_metric(y_gt, y_pred, y_prob):
    """
    Given our generated labels, compute metrics
    scoring our multi-label classification task

    Args
    ----
    y_gt   : np.ndarray of shape (N, C)
        Ground truth multi-hot labels.
    y_pred : np.ndarray of shape (N, C)
        Binary predictions (0/1) per class.
    y_prob : np.ndarray of shape (N, C)
        Predicted probabilities per class.

    Returns
    -------
    ja      : float
        Jaccard similarity averaged over samples.
    prauc   : float
        Area under precision–recall curve.
    avg_prc : float
        Mean per-sample precision (TP / predicted positives).
    avg_rec : float
        Mean per-sample recall (TP / actual positives).
    avg_f1  : float
        Mean per-sample F1 using per-sample precision/recall.
    """
    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0.0 if len(union) == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0.0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0.0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0.0)
            else:
                score.append(
                    2
                    * average_prc[idx]
                    * average_recall[idx]
                    / (average_prc[idx] + average_recall[idx])
                )
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average="macro"))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average="macro"))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(
                average_precision_score(y_gt[b], y_prob[b], average="macro")
            )
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0.0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    # roc_auc
    try:
        auc = roc_auc(y_gt, y_prob)
    except:
        auc = 0.0

    # These metrics aren't returned
    # precision
    # p_1 = precision_at_k(y_gt, y_prob, k=1)
    # p_3 = precision_at_k(y_gt, y_prob, k=3)
    # p_5 = precision_at_k(y_gt, y_prob, k=5)
    # # macro f1
    # f1 = f1(y_gt, y_pred)
    # precision
    prauc = precision_auc(y_gt, y_prob)
    # jaccard
    ja = jaccard(y_gt, y_pred)
    # pre, recall, f1
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


def ddi_rate_score(record, path):
    """
    Compute DDI rate in predictions.

    Parameters
    ----------
    record : list[list[list[int]]]
       record[p][v] = list of med codes predicted for visit v of patient p.
    path : str
       Path to ddi_A_iii.pkl adjacency matrix.

    Returns
    -------
    ddi_rate : float
    """
    ddi_A = dill.load(open(path, "rb"))
    all_pairs = 0
    ddi_pairs = 0

    for patient in record:
        for adm in patient:
            meds = adm
            for i, mi in enumerate(meds):
                for j, mj in enumerate(meds):
                    if j <= i:
                        continue
                    all_pairs += 1
                    if ddi_A[mi, mj] == 1 or ddi_A[mj, mi] == 1:
                        ddi_pairs += 1

    if all_pairs == 0:
        return 0.0

    return ddi_pairs / all_pairs


from satsolver import *


def Post_DDI(pred_result, ddi_pair, ehr_train_pair):
    """
    Now we apply the 2-SAT post-processing used in the DrugRec paper.

    This removes medication pairs forbidden by:
    - DDI adjacency matrix
    - but allowed if seen together often in EHR (ehr_train_pair)

    Parameters
    ----------
    pred_result     : (seq, voc_size) float numpy array of probabilities
    ddi_pair        : list of (i,j) pairs where DDI=1
    ehr_train_pair  : list of real-world allowed medication pairs

    Returns
    -------
    post_result : np.ndarray same shape as pred_result, binary
    """
    post_result, y_pred = np.zeros_like(pred_result), np.zeros_like(pred_result)
    y_pred[pred_result >= 0.5] = 1
    for k in range(pred_result.shape[0]):
        pred_idx = np.nonzero(y_pred[k])[0]                   # predicted meds for visit t

        # Map each predicted med to the temporary variable index
        tmp_dict = {idx: n for n, idx in enumerate(pred_idx)}

        # Then we map variable index to the related probability
        pred_prob = {str(n): pred_result[k, idx] for n, idx in enumerate(pred_idx)}
        formula = two_cnf(pred_prob)
        ddi_list = []

        # Now we can build the 2-CNF clauses for bad medication pairs
        for i, j in ddi_pair:
            if i in pred_idx and j in pred_idx and i < j:
                if (i, j) not in ehr_train_pair:
                    # print(['~' + str(tmp_dict[i]), '~' + str(tmp_dict[j])])
                    formula.add_clause(["~" + str(tmp_dict[i]), "~" + str(tmp_dict[j])])
                    if i not in ddi_list:
                        ddi_list.append(i)
                    if j not in ddi_list:
                        ddi_list.append(j)

        # Run our 2-SAT solver
        f = two_sat_solver(formula)
        if f:
            # these are meds marked as true in solver plus meds not involved in any DDI
            pos = [list(tmp_dict.keys())[int(n)] for n, x in f.items() if x == 1] + [
                idx for idx in pred_idx if idx not in ddi_list
            ]
            post_result[k, pos] = 1
        else:
            post_result[k] = pred_result[k]

    return post_result


from scipy.stats import t


def ttest(mean1, mean2, std1, std2, n1=10, n2=10):
    mu = mean1 - mean2
    df = n1 + n2 - 2
    denominator = (
        ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) * (1 / n1 + 1 / n2) / (df)
    ) ** 0.5
    tval = mu / denominator
    if mu > 0:
        pval = (1 - t.cdf(tval, df=df)) * 2
    else:
        pval = t.cdf(tval, df=df) * 2
    return pval
