import dill
import numpy as np
from collections import defaultdict
from torch.optim import Adam
import os
import torch
import torch.nn as nn
import time
from models import DrugRec
from util import llprint, multi_label_metric, ddi_rate_score, Post_DDI
import torch.nn.functional as F
import math

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    # Older torch versions won’t have this, safe to ignore
    pass

DATA_ROOT = "../../data/output"


class Config:
    """Hard-coded configuration replacing argparse arguments."""

    # experiment settings
    Test = False  # set True to run in test/eval-only mode
    model_name = "DrugRec_mimic-iii"
    resume_path = f"saved/{model_name}/drugrec_mimic3.model"

    # optimization hyperparameters
    lr = 5e-4
    epoch = 50
    decay = 0.0
    warmup = True

    # DDI control
    target_ddi = 0.05
    w_ddi = 0.5

    # causal inference hyperparameters
    CI = True
    w_ci = 5e-4
    k_ci = 5

    # multi-visit settings
    k_mul = 3
    multivisit = True
    mulhistory = False

    # embeddings
    dim = 64
    fix_smi_rep = True

    # kept for compatibility (not used directly here)
    kp = 0.05


args = Config()
print("Config:", vars(args))


def eval(model, data_eval, voc_size, epoch, ddi_adj_path, ehr_train_pair):
    """Evaluation loop (validation / test)."""
    model.eval()
    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    ddi_pair = np.where(ddi_adj == 1)
    ddi_pair = [(ddi_pair[0][i], ddi_pair[1][i]) for i in range(len(ddi_pair[0]))]

    smm_record = []
    y_gt_list = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0.0, 0.0

    for step, patient in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

        # model returns logits for each visit in the patient sequence
        target_output, _ = model(patient, "eval")

        for adm_idx, adm in enumerate(patient):
            # ground truth multi-hot
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[-1]] = 1
            y_gt.append(y_gt_tmp)

            # prediction probabilities
            target_output_ = (
                torch.sigmoid(target_output[[adm_idx]]).detach().cpu().numpy()[0]
            )
            y_pred_prob.append(target_output_)

            # copy for post-processing
            y_pred_tmp = target_output_.copy()

            # 2-SAT DDI post-processing
            y_pred_tmp = Post_DDI(
                y_pred_tmp.reshape(1, -1), ddi_pair, ehr_train_pair
            ).reshape(-1)

            # threshold to get binary predictions
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            # label indices
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))

            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        y_gt_list.append(y_gt)

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)

        llprint(f"\rtest step: {step} / {len(data_eval)}")

    ddi_rate = ddi_rate_score(smm_record, path=ddi_adj_path)

    llprint(
        "\nDDI Rate: {:.4f}, Jaccard: {:.4f},  PRAUC: {:.4f}, "
        "AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}\n".format(
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            med_cnt / visit_cnt,
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )


def main():
    # -------------------
    #  Paths (hard-coded)
    # -------------------
    data_path = os.path.join(DATA_ROOT, "records_final_iii.pkl")
    voc_path = os.path.join(DATA_ROOT, "voc_iii_sym1_mulvisit.pkl")
    ddi_adj_path = os.path.join(DATA_ROOT, "ddi_A_iii.pkl")

    input_smiles_path = os.path.join(DATA_ROOT, "input_smiles_init_rep_iii.pkl")
    sym_count_path = os.path.join(DATA_ROOT, "sym_count_iii.pkl")
    sym_input_ids_path = os.path.join(DATA_ROOT, "sym_txt_input_ids_iii.pkl")
    sym2idx_path = os.path.join(DATA_ROOT, "sym2idx_iii.pkl")
    sym_comatrix_path = os.path.join(DATA_ROOT, "sym_comatrix_iii.npy")

    save_dir = os.path.join("saved", args.model_name)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------
    #  Load data/vocabs
    # -------------------
    data = dill.load(open(data_path, "rb"))
    voc = dill.load(open(voc_path, "rb"))
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]

    # split train/val/test
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    data_eval = data[split_point + eval_len :]

    voc_size = (
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word),
    )

    # -------------------
    #  DDI adjacency
    # -------------------
    ddi_adj = dill.load(open(ddi_adj_path, "rb"))

    # co-occurrence matrix for EHR drug pairs
    drug_co_train = np.zeros((voc_size[-1], voc_size[-1]))
    for patient in data_train:
        for adm in patient:
            med_set = adm[-1]
            for med_i in med_set:
                for med_j in med_set:
                    if med_j <= med_i:
                        continue
                    drug_co_train[med_i, med_j] += 1

    drug_train_pair = np.zeros_like(drug_co_train)
    drug_train_pair[drug_co_train >= 1000] = 1
    ehr_train_pair = np.where(drug_train_pair == 1)
    ehr_train_pair = [
        (ehr_train_pair[0][i], ehr_train_pair[1][i])
        for i in range(len(ehr_train_pair[0]))
    ]

    # -------------------
    #  Symptom info & SMILES embeddings
    # -------------------
    input_smiles_init_rep = dill.load(open(input_smiles_path, "rb"))
    sym_count = dill.load(open(sym_count_path, "rb"))
    sym_input_ids = dill.load(open(sym_input_ids_path, "rb"))
    sym2idx = dill.load(open(sym2idx_path, "rb"))
    sym_comatrix = np.load(sym_comatrix_path)
    sym_information = [sym_count, sym2idx, sym_comatrix, sym_input_ids]

    # -------------------
    #  Build DrugRec_all model
    # -------------------
    model = DrugRec(
        args,
        sym_information,
        ddi_adj,
        input_smiles_init_rep,
        emb_dim=args.dim,
        device=device,
    )
    print(model)

    model.to(device=device)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model, dim=0)

    # -------------------
    #  Test-only mode
    # -------------------
    if args.Test:
        model.load_state_dict(
            torch.load(open(args.resume_path, "rb"), map_location=torch.device("cpu"))
        )
        model.to(device=device)
        tic = time.time()

        result = []
        for _ in range(10):
            test_sample = np.random.choice(
                data_test, round(len(data_test) * 0.8), replace=True
            )
            with torch.set_grad_enabled(False):
                ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
                    model, test_sample, voc_size, 0, ddi_adj_path, ehr_train_pair
                )
                result.append([ddi_rate, ja, avg_f1, prauc, avg_med])

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\\pm$ {:.4f} & ".format(m, s)
        print(outstring)
        print(f"test time: {time.time() - tic}")
        return

    # -------------------
    #  Training setup
    # -------------------
    optimizer = Adam(list(model.parameters()), lr=args.lr, weight_decay=args.decay)

    history = defaultdict(list)
    best_ja_epoch = best_auc_epoch = best_f1_epoch = 0
    best_ja = best_auc = best_f1 = 0

    EPOCH = args.epoch

    # LR scheduler with warmup + cosine decay
    if args.warmup:
        warmup_epoch = int(EPOCH * 0.06)
        iter_per_epoch = len(data_train)

        def warm_up_with_cosine_lr(step_idx):
            if step_idx <= (warmup_epoch * iter_per_epoch):
                return step_idx / (warmup_epoch * iter_per_epoch)
            else:
                progress = (step_idx - warmup_epoch * iter_per_epoch) / (
                    (EPOCH - warmup_epoch) * iter_per_epoch
                )
                return 0.5 * (math.cos(progress * math.pi) + 1)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warm_up_with_cosine_lr
        )

    learning_rates = []

    # precompute pair index mapping
    k = 0
    pair_dict = {}
    pair_size = int(voc_size[2] * (voc_size[2] - 1) / 2)
    trans_p1, trans_p2 = torch.zeros(pair_size), torch.zeros(pair_size)
    for i in range(voc_size[2]):
        for j in range(voc_size[2]):
            if j > i:
                pair_dict[(i, j)] = k
                trans_p1[k], trans_p2[k] = i, j
                k += 1
    assert len(pair_dict) == pair_size
    trans_pair_iii = (trans_p1.long(), trans_p2.long())

    # -------------------
    #  Training loop
    # -------------------
    global_step = 0
    for epoch in range(EPOCH):
        tic = time.time()
        print(f"\nepoch {epoch} --------------------------")

        model.train()
        for step, patient in enumerate(data_train):
            batch_size = len(patient)
            loss_bce_target = np.zeros((batch_size, voc_size[2]))
            loss_multi_target = np.full((batch_size, voc_size[2]), -1)
            loss_bce_pair_target = np.zeros(
                (batch_size, int(voc_size[2] * (voc_size[2] - 1) / 2))
            )

            # build targets for each admission in this patient sequence
            for idx, adm in enumerate(patient):
                med_list = adm[-1]
                loss_bce_target[idx, med_list] = 1

                # for multilabel_margin_loss: list of labels per row
                for i, item in enumerate(med_list):
                    loss_multi_target[idx][i] = item

                # pairwise co-occurrence targets
                for i in range(len(med_list)):
                    for j in range(len(med_list)):
                        if j > i:
                            pair_idx = pair_dict[(i, j)]
                            loss_bce_pair_target[idx, pair_idx] = 1

            if args.CI:
                result, neg_pred_prob, loss_ddi, pred_prob, pred_prob0 = model(
                    patient, "train"
                )
                # CI loss from paper: encourage correct direction of effect
                sign = torch.sign(torch.FloatTensor(loss_bce_target - 0.5).to(device))
                loss_ci = -torch.log(
                    torch.sigmoid(sign * (pred_prob - pred_prob0))
                ).mean()
            else:
                result, neg_pred_prob, loss_ddi = model(patient, "train")

            # map pair outputs
            result_pair = torch.zeros(
                batch_size, int(voc_size[2] * (voc_size[2] - 1) / 2)
            )
            for i in range(batch_size):
                result_pair[i] = neg_pred_prob[i][trans_pair_iii]

            loss_bce = F.binary_cross_entropy_with_logits(
                result, torch.FloatTensor(loss_bce_target).to(device)
            )
            loss_multi = F.multilabel_margin_loss(
                torch.sigmoid(result),
                torch.LongTensor(loss_multi_target).to(device),
            )
            loss_bce_pair = F.binary_cross_entropy(
                result_pair.to(device),
                torch.FloatTensor(loss_bce_pair_target).to(device),
            )

            # compute DDI rate for current prediction (per patient)
            result_sig = torch.sigmoid(result).detach().cpu().numpy()
            result_sig[result_sig >= 0.5] = 1
            result_sig[result_sig < 0.5] = 0
            y_label = np.where(result_sig == 1)[0]
            current_ddi_rate = ddi_rate_score([[y_label]], path=ddi_adj_path)

            if current_ddi_rate <= args.target_ddi:
                loss = loss_bce + 0.1 * loss_multi + loss_bce_pair
            else:
                loss = (
                    loss_bce + 0.1 * loss_multi + args.w_ddi * loss_ddi + loss_bce_pair
                )

            if args.CI:
                loss = loss + args.w_ci * loss_ci

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if args.warmup:
                scheduler.step()
                learning_rates.append(scheduler.get_last_lr()[0])
            global_step += 1

            if step % 10 == 0:
                if args.CI:
                    llprint(
                        "\rtraining step: {} / {}, loss: {} (loss_ci: {}, loss_ddi: {}, loss_bce_pair: {}), time: {}s".format(
                            step,
                            len(data_train),
                            str(loss.item())[:7],
                            str(loss_ci.item())[:7],
                            str(loss_ddi.item())[:7],
                            str(loss_bce_pair.item())[:7],
                            str(time.time() - tic)[:5],
                        )
                    )
                else:
                    llprint(
                        "\rtraining step: {} / {}, loss: {} (loss_ddi: {}), time: {}s".format(
                            step,
                            len(data_train),
                            str(loss.item())[:7],
                            str(loss_ddi.item())[:7],
                            str(time.time() - tic)[:5],
                        )
                    )

        # -------------------
        #  Epoch-end evaluation
        # -------------------
        print(args.model_name)
        tic2 = time.time()
        with torch.set_grad_enabled(False):
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
                model, data_eval, voc_size, epoch, ddi_adj_path, ehr_train_pair
            )
            print(
                "training time: {}, test time: {}".format(
                    time.time() - tic, time.time() - tic2
                )
            )

            history["ja"].append(ja)
            history["ddi_rate"].append(ddi_rate)
            history["avg_p"].append(avg_p)
            history["avg_r"].append(avg_r)
            history["avg_f1"].append(avg_f1)
            history["prauc"].append(prauc)
            history["med"].append(avg_med)

            # save checkpoint
            ckpt_path = os.path.join(
                save_dir,
                "Epoch_{}_TARGET_{:.2f}_JA_{:.4f}_AUC_{:.4f}_F1_{:.4f}_DDI_{:.4f}.model".format(
                    epoch, args.target_ddi, ja, prauc, avg_f1, ddi_rate
                ),
            )
            torch.save(model.state_dict(), open(ckpt_path, "wb"))

            # moving-average best metrics
            if epoch >= 5:
                print(
                    "ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(
                        np.mean(history["ddi_rate"][-5:]),
                        np.mean(history["med"][-5:]),
                        np.mean(history["ja"][-5:]),
                        np.mean(history["avg_f1"][-5:]),
                        np.mean(history["prauc"][-5:]),
                    )
                )

                if best_ja < np.mean(history["ja"][-5:]):
                    best_ja_epoch = epoch
                    best_ja = ja
                print(f"best_ja_epoch: {best_ja_epoch}")

                if best_auc < np.mean(history["prauc"][-5:]):
                    best_auc_epoch = epoch
                    best_auc = prauc
                print(f"best_auc_epoch: {best_auc_epoch}")

                if best_f1 < np.mean(history["avg_f1"][-5:]):
                    best_f1_epoch = epoch
                    best_f1 = avg_f1
                print(f"best_f1_epoch: {best_f1_epoch}")

    dill.dump(learning_rates, open(f"saved/{args.model_name}/learning_rates", "wb"))


if __name__ == "__main__":
    main()
    print(args.model_name)
