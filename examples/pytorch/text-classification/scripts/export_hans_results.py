import os
import shutil
import torch
import json
import pandas as pd

runs_path = "../runs"
runs = os.listdir(runs_path)

model_names = {'google/bert_uncased_L-4_H-256_A-4': 'MiniBert', 'google/bert_uncased_L-2_H-128_A-2': 'TinyBert', 'bert-base-uncased': 'Bert'}

hans_data = []

for run in runs:
    run_path = os.path.join(runs_path, run)
    if os.path.isfile(os.path.join(run_path, 'evaluate_heuristic_output.txt')):
        os.mkdir(os.path.join(run_path, 'hans'))
        shutil.move(os.path.join(run_path, 'evaluate_heuristic_output.txt'), os.path.join(run_path, 'hans/evaluate_heuristic_output.txt'))
        shutil.move(os.path.join(run_path, 'hans_predictions.txt'), os.path.join(run_path, 'hans/hans_predictions.txt'))

    if not os.path.isdir(os.path.join(run_path, 'hans')):
        print(f'run {run} does not have HANS evaluation results')
        continue


    row = {}
    model_config = json.load(open(os.path.join(run_path, 'config.json'), 'r'))
    row['model_name'] = model_config['_name_or_path']
    if row['model_name'] in model_names:
        row['model_name'] = model_names[row['model_name']]
    training_args = torch.load(os.path.join(run_path, 'training_args.bin'))
    row['weak_model_name'] = training_args.weak_model_path
    row['wl'] = training_args.weak_model_layer
    row['ml'] = training_args.regularized_layers
    row['reg_lambda'] = training_args.regularization_lambda
    row['reg_method'] = training_args.regularization_method
    row['batch_size'] = training_args.per_device_train_batch_size
    row['warmup_steps'] = training_args.warmup_steps
    row['weight_decay'] = training_args.weight_decay
    row['learning_rate'] = training_args.learning_rate
    eval_results = json.load(open(os.path.join(run_path, 'eval_results.json')))
    row['eval_sim_loss'] = eval_results.get('eval_sim_loss', None)
    row['eval_acc'] = eval_results['eval_accuracy']
    results_data = open(os.path.join(run_path, 'hans/evaluate_heuristic_output.txt'), 'r').readlines()
    results_dict = {'entailment': 0, 'non-entailment': 0}
    start_index = 1
    for k in results_dict.keys():
        sum = 0
        for i in range(start_index, start_index + 3):
            sum += float(results_data[i].strip().split(":")[1])
        sum /= 3
        results_dict[k] = sum
        start_index = 6

    row.update(results_dict)
    hans_data.append(row)

if not os.path.isfile('results.csv'):
    pd.DataFrame(hans_data).to_csv('results.csv',index=False)
else:
    print('Results file exists. stopping to prevent overwrite')