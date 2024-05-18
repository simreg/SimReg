from sklearn.metrics import f1_score, precision_score, precision_recall_fscore_support
from transformers.trainer_utils import EvalPrediction
import numpy as np
from numpy import ndarray
from scipy.special import softmax
from datasets import load_metric
from arguments_classes import DataTrainingArguments


def hans_compute_metrics(p: EvalPrediction, old_labels_order=False):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    if old_labels_order:
        preds[preds == 2] = 0
    else:
        preds[preds == 2] = 1
    return {"accuracy": simple_accuracy(preds, p.label_ids)}


def flip_predictions_poe(preds):
    cont_col = preds[:, 0].clone()
    ent_col = preds[:, 1].clone()
    nuet_col = preds[:, 2].clone()
    preds[:, 0] = ent_col
    preds[:, 1] = nuet_col
    preds[:, 2] = cont_col
    return preds


def prefix_keys(dictionary: dict, prefix):
    keys = list(dictionary.keys())
    for k in keys:
        dictionary[f"{prefix}_{k}"] = dictionary.pop(k)
    return dictionary


def simple_accuracy(preds: ndarray, labels: ndarray):
    return float((preds == labels).mean())


# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
def get_metrics_function(is_regression, data_args: DataTrainingArguments, synthetic_bias_indices: dict = None, poe_adustment=False):
    # Get the metric function
    if data_args.task_name is not None and data_args.task_name not in ('fever',):
        metric = load_metric("glue", data_args.task_name, )
    else:
        metric = load_metric("accuracy")

    if synthetic_bias_indices is not None:
        biased_indices = synthetic_bias_indices['aligned_indices']
        anti_biased_indices = synthetic_bias_indices['misaligned_indices']

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        # contradiction, entailment, neutral - poe
        # entailment, neutral, contradiction. -- current
        if poe_adustment:
            preds = flip_predictions_poe(preds)

        probs = softmax(preds, axis=-1)
        entropy = float(np.exp((-probs * np.log(probs)).sum(axis=-1).mean()))
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result: dict = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()

            if data_args.task_name == 'qqp':
                _, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, labels=[0, 1])
                duplicate_results = {'recall': recall[1]}
                non_duplicate_results = {'recall': recall[0], 'f1': f1[0]}
                result.update(prefix_keys(duplicate_results, 'dup'))
                result.update(prefix_keys(non_duplicate_results, 'non_dup'))

            result['entropy'] = entropy
            if synthetic_bias_indices is not None:
                # compute acc on each subset
                acc_result = (preds == p.label_ids).astype(np.float32)
                result['synthetic_bias_acc'] = acc_result[biased_indices].mean().item()
                result['synthetic_anti_bias_acc'] = acc_result[anti_biased_indices].mean().item()
            return result
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
    return compute_metrics

