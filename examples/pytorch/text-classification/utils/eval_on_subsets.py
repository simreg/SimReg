from glob import glob
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# models = glob('runs/bert_linear_cka_11_3_lambda_*_batch_128')
# models = glob('runs/bert_linear_cka_11_3_lambda_1_batch_64')
models = glob('runs/*')
# models = glob('runs/bert_reproduce_128_batch')
# models = glob('runs/bert_reproduce_32_batch')
for model_dir in models:
    print(model_dir)
    ckpts = glob(f"{model_dir}/checkpoint-*")
    ckpts.sort(key=lambda x: int(x.split("/")[-1].split("-")[1]))
    results_dct = {'HANS_ent': dict(), 'HANS_non_ent':dict()}
    for ckpt in ckpts:
        result_files = glob(f"{ckpt}/evaluation/*.json")
        for result in result_files:
            result_file = os.path.basename(result)
            result_name = result_file.split(".")[0]
            if result_name == 'all_results':
                continue

            if result_name not in results_dct:
                results_dct[result_name] = dict()
            with open(result, 'r') as f:
                results_dct[result_name][int(ckpt.split("/")[-1].split("-")[1])] = json.load(f)['eval_accuracy']
        results_data = open(os.path.join(ckpt, 'hans/evaluate_heuristic_output.txt'), 'r').readlines()
        hans_dict = {'entailment': 0, 'non-entailment': 0}
        start_index = 1
        for k in hans_dict.keys():
            sum = 0
            for i in range(start_index, start_index + 3):
                sum += float(results_data[i].strip().split(":")[1])
            sum /= 3
            hans_dict[k] = sum
            start_index = 6
        results_dct['HANS_ent'][int(ckpt.split("/")[-1].split("-")[1])] = hans_dict['entailment']
        results_dct['HANS_non_ent'][int(ckpt.split("/")[-1].split("-")[1])] = hans_dict['non-entailment']

    if len(ckpts) == 0:
        if os.path.isfile(os.path.join(model_dir, 'evaluation_ckpts.png')):
            os.remove(os.path.join(model_dir, 'evaluation_ckpts.png'))
        continue
    plt.clf()
    plt.figure(figsize=(13, 10))
    for k in results_dct.keys():
        sns.lineplot(x=results_dct[k].keys(), y=results_dct[k].values())

    plt.legend(list(results_dct.keys()))
    plt.xlabel('training step')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'evaluation_ckpts.png'))



