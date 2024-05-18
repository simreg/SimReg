import torch
from torch import Tensor
from torch.nn.functional import normalize
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, set_seed, default_data_collator
from models.simple_models import SimpleMLP, SimpleTokenizer
from run_glue import setup_args, load_datasets, get_datasets, build_trainer, do_eval, preprocess_raw_datasets, get_preprocess_function
from os import path
from typing import Dict, List


class ParamMeanWatch(TrainerCallback):
    hist_ref: Dict[str, List[int]] = None
    tokens_sim_hist: Dict[str, List[int]] = None

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: SimpleMLP, **kwargs):
        vocab = {'a': 1, 'b': 2, 'c': 3}
        reverse_vocab = ['PAD', 'a', 'b', 'c']
        token_vecs = []
        with torch.no_grad():
            for c, token_id in vocab.items():
                if c not in self.hist_ref:
                    self.hist_ref[c] = []
                token_vec: Tensor = model.embeddings.weight[token_id]
                token_vecs.append(token_vec)
                self.hist_ref[c].append(token_vec.norm().item())
            tokens_mat: Tensor = torch.stack(token_vecs)
            tokens_mat = normalize(tokens_mat)
            sim_mat = tokens_mat @ tokens_mat.t()
            for i in range(1, 4):
                for j in range(i + 1, 4):
                    key = f'{reverse_vocab[i]}-{reverse_vocab[j]}'
                    if key not in self.tokens_sim_hist:
                        self.tokens_sim_hist[key] = []
                    self.tokens_sim_hist[key].append(sim_mat[i - 1, j - 1].item())


def main():
    data_args, last_checkpoint, model_args, training_args = setup_args()
    # Set seed before initializing model.
    set_seed(training_args.seed)
    raw_datasets = load_datasets(data_args, model_args, training_args)
    simple_tokenizer = SimpleTokenizer()
    prepcoessing_func = get_preprocess_function(simple_tokenizer, 'premise', 'hypothesis', max_seq_length=None, padding=0)
    bias_indices, raw_datasets = preprocess_raw_datasets(data_args, raw_datasets, training_args, num_labels=2,
                                                         preprocessing_func=prepcoessing_func, sentence1_key=None)
    # data_args, raw_datasets, training_args, hans_preprocess_func = None, tokenizer: PreTrainedTokenizer = None, lexical_bias_settings = False, biased_samples = None
    eval_dataset, predict_dataset, train_dataset = get_datasets(data_args, raw_datasets, training_args)
    if training_args.weak_model_path is None and model_args.model_name_or_path != 'baseline':
        # assuming that we are training the weak model, using 0 hidden layers
        model = SimpleMLP(in_dim=10, hidden_dim=10, n_layers=1, out_dim=2)
    else:
        model = SimpleMLP(in_dim=10, hidden_dim=10, n_layers=2, out_dim=2)


    weak_model = None
    if training_args.weak_model_path is not None:
        if 'weak' in training_args.weak_model_path:
            weak_model = SimpleMLP(in_dim=10, hidden_dim=10, n_layers=1, out_dim=2)
        else:
            weak_model = SimpleMLP(in_dim=10, hidden_dim=10, n_layers=2, out_dim=2)

        wm_state_dct_path = path.join(training_args.weak_model_path, "pytorch_model.bin")
        weak_model.load_state_dict(torch.load(wm_state_dct_path))
    print("main model:")
    print(model)
    print("weak model:")
    print(weak_model)

    trainer = build_trainer(None, data_args, default_data_collator, eval_dataset, None, model, None, train_dataset, training_args, weak_model)
    if training_args.regularization_method is not None:
        mean_callback = ParamMeanWatch()
        # Brook it to clean the code, check in the git history if you want to re-run it.
        mean_callback.hist_ref = trainer.param_mean_hist
        mean_callback.tokens_sim_hist = trainer.tokens_sim_hist
        trainer.add_callback(mean_callback)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        do_eval(data_args, eval_dataset, raw_datasets, trainer)


if __name__ == '__main__':
    main()
