import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

"""
this script plots HANS performance over checkpoints of a certain model
"""
model_name = "new_bert_linear_cka_10_3_lambda_1"
base_dir = f"../runs/{model_name}"
ckpts = os.listdir(base_dir)
ckpts = list(filter(lambda x: x.startswith("checkpoint-"), ckpts))
lines = {'entailment': [[], [], []], 'non-entailment': [[], [], []]}
ckpts.sort(key=lambda x: int(x.split("-")[1]))
for ckpt in ckpts:
    results_data = open(os.path.join(base_dir, f"{ckpt}/hans/evaluate_heuristic_output.txt"), 'r').readlines()
    results_dict = {'entailment': 0, 'non-entailment': 0}
    start_index = 1
    for k in results_dict.keys():
        sum = 0
        for i in range(start_index, start_index + 3):
            c = float(results_data[i].strip().split(":")[1])
            sum += c
            lines[k][i - start_index].append(c)
        sum /= 3
        results_dict[k] = sum
        start_index = 6
    # entailment.append(results_dict['entailment'])
    # non_ent.append(results_dict['non-entailment'])

plt.clf()
legends = []
hurestics = ['lexical overlap', 'subsequence', 'constituent']
for k in lines.keys():
    for i in range(len(lines[k])):
        plt.plot(lines[k][i])
        legends.append(f"{k}_{hurestics[i]}")

plt.legend(legends)
plt.savefig(f'{model_name}_ckpts_hans.png')
