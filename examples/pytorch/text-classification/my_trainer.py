import inspect
import logging
from collections.abc import Mapping
from typing import Dict, List, Optional

import datasets
import torch.nn.modules.module
from torch.utils.data import DataLoader
from transformers.trainer import Trainer
from transformers import TrainingArguments, is_datasets_available
from torch.nn.functional import normalize
from transformers.pipelines.base import Dataset
from transformers.trainer_utils import AggregationStrategy, BiasSamplingStrategy

from models.lexical_bias_bert import ClarkLexicalBiasModel
from utils.bias_sampler import BiasBatchSampler
from models import BertWithLexicalBiasModel
from utils.misc import EvaluationSet


def get_activation(activations_storage, i, append=False):
    def hook(model, input, output):
        if isinstance(output, tuple):
            output = output[0]
        if append:
            activations_storage.append(output)
        else:
            activations_storage[i] = output

    return hook


def off_diag_mean(input_tensor):
    return input_tensor.sum() / (input_tensor.shape[0] * (input_tensor.shape[1] - 1))


def get_model_signature(model):
    return inspect.signature(model.forward).parameters.keys()


class BaseTrainer(Trainer):
    def __init__(self, evaluation_sets: List[EvaluationSet] = None, **kwargs):
        super(BaseTrainer, self).__init__(**kwargs)
        self.logging_mode = 'train'
        self.xe_loss_log = {'train': [], 'eval': []}
        self.evaluation_sets = evaluation_sets if evaluation_sets is not None else []
        self.tokens_sim_hist = {}
        assert self.args.n_gpu == 1, "This code have not been tested on multi-GPU settings."

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            compute_speed_metrics: bool = True
    ) -> Dict[str, float]:
        prev_logging_mode = self.logging_mode
        self.logging_mode = metric_key_prefix
        res = super(BaseTrainer, self).evaluate(eval_dataset, ignore_keys, metric_key_prefix, compute_speed_metrics=compute_speed_metrics)
        if self.is_in_train and eval_dataset is None:
            for evaluation_set in self.evaluation_sets:
                prev_metrics_func = self.compute_metrics
                if evaluation_set.metrics_func is not None:
                    self.compute_metrics = evaluation_set.metrics_func
                self.logging_mode = evaluation_set.logging_mode
                super(BaseTrainer, self).evaluate(eval_dataset=evaluation_set.set, ignore_keys=ignore_keys,
                                                  compute_speed_metrics=False, metric_key_prefix=self.logging_mode)
                self.compute_metrics = prev_metrics_func

        self.logging_mode = prev_logging_mode
        return res

    def _log_feature(self, logs, feature_name, log_name):
        if hasattr(self, feature_name) and len(getattr(self, feature_name)[self.logging_mode]) > 0:
            feature = getattr(self, feature_name)
            logs[log_name] = sum(feature[self.logging_mode]) / len(feature[self.logging_mode])
            feature[self.logging_mode] = []

    def log_loss(self, xe_loss: torch.Tensor, sim_loss: torch.Tensor = None):
        # log loss terms individually
        if self.logging_mode is not None:
            if hasattr(self, 'reg_loss_log') and sim_loss is not None:
                if self.logging_mode not in self.reg_loss_log:
                    self.reg_loss_log[self.logging_mode] = []

                self.reg_loss_log[self.logging_mode].append(sim_loss.item())

            if self.logging_mode not in self.xe_loss_log:
                self.xe_loss_log[self.logging_mode] = []
            self.xe_loss_log[self.logging_mode].append(xe_loss.item())

    def log(self, logs) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        prefix = f'{self.logging_mode}_' if self.logging_mode != 'train' else ''

        self._log_feature(logs, 'reg_loss_log', f'{prefix}sim_loss')
        self._log_feature(logs, 'xe_loss_log', f'{prefix}xe_loss')
        self._log_feature(logs, 'effective_batch_size', f'{prefix}_effective_batch_size')

        if hasattr(self, 'sim_hist') and self.logging_mode in self.sim_hist:
            logging.info(f"Similarity among layers (mode={self.logging_mode}):")
            for k in self.sim_hist[self.logging_mode].keys():
                if len(self.sim_hist[self.logging_mode][k]) > 0:
                    logging.info(f"{k}: {sum(self.sim_hist[self.logging_mode][k]) / len(self.sim_hist[self.logging_mode][k])}")
                else:
                    logging.info(f"{k}: empty!")
                self.sim_hist[self.logging_mode][k].clear()

        if self.logging_mode == 'train':
            for k, hist in self.tokens_sim_hist.items():
                if len(hist) == 0:
                    continue
                logs[f'{k}_sim'] = sum(hist) / len(hist)
                self.tokens_sim_hist[k] = []

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = super(BaseTrainer, self).compute_loss(model=model, inputs=inputs, return_outputs=return_outputs)
        self.log_loss(loss[0] if isinstance(loss, tuple) else loss)
        if self.args.main_task_lambda != 1.0 and self.logging_mode == "train":
            if isinstance(loss, tuple):
                loss = (loss[0] * self.args.main_task_lambda,) + loss[1:]
            else:
                loss = loss * self.args.main_task_lambda
        return loss

    def _prepare_input(self, data):
        """
        Prepares one :obj:`data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.args.device)
            if self.deepspeed and data.dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
            return data.to(**kwargs)
        return data


class SimRegTrainer(BaseTrainer):
    def __init__(
            self,
            model,
            weak_models=None,
            bias_indices=None,
            args: TrainingArguments = None,
            **kwargs
    ):
        super(SimRegTrainer, self).__init__(model=model, args=args, **kwargs)
        self.bias_sampler: BiasBatchSampler = None
        self.sim_hist = dict()
        self.regularized_layers = args.regularized_layers
        self.weak_models = weak_models
        self.weak_models_layers = args.weak_models_layers
        # assert args.regularized_layers is not None and len(args.regularized_layers) == len(args.weak_model_layers)
        # initialized to empty lists to ensure passing them by-reference.
        self.activations = [None] * len(self.regularized_layers)
        self.weak_activations = [[None] * len(fw_layers) for fw_layers in self.weak_models_layers]
        self.reg_lambda = args.regularization_lambda

        # hook layers
        assert model is not None
        if model is not None:
            for i in range(len(self.regularized_layers)):
                model.get_submodule(self.regularized_layers[i]).register_forward_hook(get_activation(self.activations, i))
            self.main_signature = set(get_model_signature(model))
        if weak_models is not None:
            for fw_idx, fw in enumerate(weak_models):
                for i in range(len(self.weak_models_layers[fw_idx])):
                    fw.get_submodule(self.weak_models_layers[fw_idx][i]).register_forward_hook(get_activation(self.weak_activations[fw_idx], i))

        self.reg_loss_log = {'train': [], 'eval': []}
        if self.args.regularize_only_biased:
            logging.info("Regularizing only biased samples")
            assert bias_indices is not None
        self.bias_indices = bias_indices
        # assigned by outside code
        self.synthetic_bias_settings = False

    def get_train_dataloader(self) -> DataLoader:
        if self.args.regularize_only_biased and self.args.bias_sampling_strategy != BiasSamplingStrategy.NONE:
            train_dataset = self.train_dataset
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                train_dataset = self._remove_unused_columns(train_dataset, description="training")
            bias_sampler = BiasBatchSampler(self.bias_indices, len(self.train_dataset), self.args.train_batch_size, False,
                                            self.args.bias_sampling_strategy)
            self.bias_sampler = bias_sampler
            return DataLoader(
                train_dataset,
                batch_sampler=bias_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            return super(SimRegTrainer, self).get_train_dataloader()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        batch_bias_indices = None

        if self.bias_sampler is None and self.args.regularize_only_biased and self.logging_mode == 'train':
            assert "idx" in inputs
            assert self.bias_indices is not None
            from numpy import intersect1d
            indices = inputs.pop('idx').cpu().numpy()
            _, batch_bias_indices, _ = intersect1d(indices, self.bias_indices, return_indices=True, assume_unique=True)
            batch_bias_indices = torch.from_numpy(batch_bias_indices)

        if "idx" in inputs:
            inputs.pop('idx')
        if 'id' in inputs:
            inputs.pop('id')

        # if self.activations_hook_init:
        #     logging.info(f'activations_hook activated at {os.getpid()}')
        #     if isinstance(model, DataParallel):
        #         raise NotImplementedError('Activations hook is not supported on multi-GPU yet.')
        #     for i in range(len(self.regularized_layers)):
        #         self.model.get_submodule(self.regularized_layers[i]).register_forward_hook(get_activation(self.activations, i))
        #     self.main_signature = set(get_model_signature(self.model))
        #     self.activations_hook_init = False

        main_inputs = {k: inputs[k] for k in self.main_signature.intersection(inputs.keys())}
        outputs = model(**main_inputs)

        # calculate weak_model activations
        # TODO: Maybe replace it with a saved activations, or even save the similarity structures (RSM)?
        if self.weak_models is not None:
            with torch.no_grad():
                if self.args.separate_weak_tokenization:
                    for fw_idx, fw in enumerate(self.weak_models):
                        weak_inputs = dict()
                        for k in inputs:
                            if k.startswith(f"weak_{fw_idx}_"):
                                weak_inputs[k[len(f"weak_{fw_idx}_"):]] = inputs[k]
                        fw(**weak_inputs)
                else:
                    for fw in self.weak_models:
                        fw(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.logging_mode == 'train' and self.bias_sampler is not None and not self.bias_sampler.bias_turn:
            sim_loss = None
        else:
            sim_loss = self.compute_sim_loss(inputs, indices=batch_bias_indices)

        self.log_loss(loss, sim_loss)

        if self.args.main_task_lambda != 1.0 and self.logging_mode == "train":
            if isinstance(loss, tuple):
                loss = (loss[0] * self.args.main_task_lambda,) + loss[1:]
            else:
                loss = loss * self.args.main_task_lambda

        if sim_loss is not None:
            if self.args.enforce_similarity:
                # maximize similarity instead of minimizing
                loss -= self.reg_lambda * sim_loss
            else:
                loss += self.reg_lambda * sim_loss

        return (loss, outputs) if return_outputs else loss

    def compute_sim_loss(self, inputs, indices=None):
        assert len(inputs) > 1

        sim_loss = None
        for i in range(len(self.activations)):
            main_activations = self.activations[i]
            main_activations, updated_indices = self.extract_relevant_activations(
                inputs['attention_mask'],
                main_activations,
                batch_bias_indices=indices,
                regularized_tokens=self.args.regularized_tokens[i],
                aggregation_strategy=self.args.token_aggregation_strategy[i] if self.args.token_aggregation_strategy is not None else None
            )
            for fw_idx, fw in enumerate(self.weak_models):
                weak_activations = self.weak_activations[fw_idx][i]
                if not (isinstance(fw, BertWithLexicalBiasModel) or isinstance(fw, ClarkLexicalBiasModel)):
                    weak_activations, _ = self.extract_relevant_activations(
                        inputs[f'weak_{fw_idx}_attention_mask'] if self.args.separate_weak_tokenization else inputs['attention_mask'],
                        weak_activations,
                        batch_bias_indices=indices,
                        regularized_tokens=self.args.regularized_tokens[i],
                        aggregation_strategy=self.args.token_aggregation_strategy[i] if self.args.token_aggregation_strategy is not None else None
                    )

                tmp_sm_ls = self.generic_sim_measure(main_activations, weak_activations, updated_indices, attention_mask=inputs.get('attention_mask', None))
                if self.logging_mode not in self.sim_hist:
                    self.sim_hist[self.logging_mode] = {}
                    for c in range(len(self.weak_models)):
                        self.sim_hist[self.logging_mode].update({f"fw_{c}_{k}_{v}": [] for k, v in zip(self.regularized_layers, self.weak_models_layers[c])})
                self.sim_hist[self.logging_mode][f"fw_{fw_idx}_{self.regularized_layers[i]}_{self.weak_models_layers[fw_idx][i]}"].append(tmp_sm_ls.item())
                if sim_loss is None:
                    sim_loss = tmp_sm_ls
                else:
                    sim_loss = sim_loss + tmp_sm_ls

        return sim_loss

    @staticmethod
    def extract_relevant_activations(attention_mask, main_activations, batch_bias_indices=None,
                                     regularized_tokens: str = None, aggregation_strategy: AggregationStrategy = None):
        if regularized_tokens == 'CLS':
            assert main_activations.dim() == 3
            return main_activations[:, 0, :], batch_bias_indices
        if regularized_tokens == 'all':
            # dimensions are: batch_size, n_tokens, hidden_dim
            if main_activations.dim() != 3:
                # dirty work around, that should be fixed!
                return main_activations, batch_bias_indices
            mask = attention_mask.bool()
            if aggregation_strategy == AggregationStrategy.SUM or \
                    aggregation_strategy == AggregationStrategy.MEAN:
                # Aggregate tokens
                main_activations = main_activations.clone()
                main_activations[~mask] = 0
                if aggregation_strategy == AggregationStrategy.SUM:
                    return main_activations.sum(dim=1), batch_bias_indices
                return main_activations.sum(dim=1) / mask.sum(dim=1, keepdim=True), batch_bias_indices
            else:
                # keep tokens separately, the problem here is we need to take the tokens of the biased samples and only non-PAD tokens.
                if batch_bias_indices is not None:
                    bias_mask = torch.zeros_like(mask, dtype=torch.bool)
                    bias_mask[batch_bias_indices, :] = True
                    mask = mask & bias_mask

                main_activations = main_activations.view(-1, main_activations.shape[-1])
                assert main_activations.shape[0] == mask.numel()
                main_activations = main_activations[mask.flatten()]
                # in case of separate tokens regularization, it is computationally expensive to measure sim across
                # all tokens, thats why we reduce to the biased tokens only and ignore batch_biased_indices (see LinCKA)
                return main_activations, None
        elif regularized_tokens == 'MLP' or regularized_tokens == 'none':
            # leave activations as is
            pass
        else:
            raise Exception('Illegal argument')
        return main_activations, batch_bias_indices

    @classmethod
    def sim_measure(cls, z, h, return_components=False, indices=None, attention_mask=None):
        raise NotImplemented

    def generic_sim_measure(self, z, h, indices=None, attention_mask=None):
        sim_methods = {
            'cosine': CosineSimReg,
            'cos_cor': CorSimReg,
            'abs_cos_cor': CorSimReg,
            'linear_cka': LinCKA,
        }
        if self.args.regularization_method == 'abs_cos_cor':
            kwargs = {'abs_cor': True}
        else:
            kwargs = {}
        return sim_methods[self.args.regularization_method].sim_measure(z, h, indices=indices, attention_mask=attention_mask, **kwargs)



class CosineSimReg(SimRegTrainer):
    @classmethod
    def sim_measure(cls, z, h, return_components=False,attention_mask=None, indices=None):
        # n = z.shape[0]
        z = z.view(z.shape[0], -1)
        h = h.view(h.shape[0], -1)
        z = normalize(z, p=2, dim=1)
        h = normalize(h, p=2, dim=1)
        z_RSM = z @ z.t()
        h_RSM = h @ h.t()
        return (1 - (z_RSM - h_RSM) ** 2).mean()
        # mid_op = mid_op.flatten()[1:].view(n-1, n+1)[:, :-1]
        # return (mid_op ** 2).sum()

    @classmethod
    def sim_measure_combinations(cls, z, h, bias_indices):
        mid_op = cls.sim_measure(z, h, return_components=True)
        n = mid_op.shape[0]
        # 1. sim betweeen biased an rest
        bias_rest = -off_diag_mean(mid_op[bias_indices, :])
        # 2. sim between biased and biased
        bias_bias_indices = torch.cartesian_prod(bias_indices, bias_indices)
        bias_bias_indices = (bias_bias_indices[:, 0] * n) + bias_bias_indices[:, 1]
        bias_bias = -mid_op.flatten()[bias_bias_indices].mean()
        # 3. sim between rest and rest
        off_bias_count = (mid_op.numel() - (bias_indices.shape[0] * (n - 1)))
        rest_rest = -(mid_op.sum() - mid_op[bias_indices, :].sum()) / off_bias_count
        # 4. overall sim
        overall = -off_diag_mean(mid_op)
        return {'overall': overall, 'rest_rest': rest_rest, 'bias_bias': bias_bias, 'bias_rest': bias_rest}


class CorSimReg(SimRegTrainer):
    @classmethod
    def sim_measure(cls, z, h, return_components=False, indices=None, attention_mask=None, abs_cor=False):
        assert z.shape[0] == h.shape[0]
        if indices is not None:
            z = z[indices]
            h = h[indices]
        n = z.shape[0]
        z = z.view((n, -1))
        h = h.view((n, -1))
        z = normalize(z, p=2, dim=1)
        h = normalize(h, p=2, dim=1)
        z_RSM = torch.matmul(z, z.t()).flatten()[1:].view(n-1, n+1)[:, :-1]
        h_RSM = torch.matmul(h, h.t()).flatten()[1:].view(n-1, n+1)[:, :-1]
        z_RSM = z_RSM - z_RSM.mean()
        h_RSM = h_RSM - h_RSM.mean()
        res = (z_RSM * h_RSM).sum() / (z_RSM.norm() * h_RSM.norm())
        return res.abs() if abs_cor else res

    @classmethod
    def sim_measure_combinations(cls, z, h, bias_indices):
        mid_op, z_RSM, h_RSM = cls.sim_measure(z, h, return_components=True)
        h_RSM = h_RSM ** 2
        z_RSM = z_RSM ** 2
        n = mid_op.shape[0]
        # TODO: consider subtracting the length of main diagonal from computation, since it only zeros.
        # 1. sim betweeen biased an rest
        bias_rest = off_diag_mean(mid_op[bias_indices, :]) / (torch.sqrt(off_diag_mean(h_RSM[bias_indices, :]) * off_diag_mean(z_RSM[bias_indices, :])))
        # 2. sim between biased and biased
        bias_bias_indices = torch.cartesian_prod(bias_indices, bias_indices)
        bias_bias_indices = (bias_bias_indices[:, 0] * n) + bias_bias_indices[:, 1]
        bias_bias_mean = mid_op.flatten()[bias_bias_indices].mean()
        bias_bias = bias_bias_mean / torch.sqrt(h_RSM.flatten()[bias_bias_indices].mean() * z_RSM.flatten()[bias_bias_indices].mean())
        # 3. sim between rest and rest
        off_bias_count = (mid_op.numel() - (bias_indices.shape[0] * (n-1)))
        rest_rest = ((mid_op.sum() - mid_op[bias_indices, :].sum()) / off_bias_count) / torch.sqrt(((h_RSM.sum() - h_RSM[bias_indices, :].sum()) / off_bias_count) * ((z_RSM.sum() - z_RSM[bias_indices, :].sum()) / off_bias_count))
        # 4. overall sim
        overall = off_diag_mean(mid_op) / torch.sqrt(off_diag_mean(h_RSM) * off_diag_mean(z_RSM))
        return {'overall': overall, 'rest_rest': rest_rest, 'bias_bias': bias_bias, 'bias_rest': bias_rest}


class CKAReg(SimRegTrainer):
    @staticmethod
    def center(z, unbiased=True):
        """
        debiased version of centering (To calculate debiased HSIC estimator)
        """
        n = z.shape[0]
        if unbiased:
            z[range(n), range(n)] *= 0
            means = torch.sum(z, dim=0, keepdim=True) / (n - 2)
            means -= torch.sum(means) / (2 * (n - 1))
            z -= means.t()
            z -= means
            z[range(n), range(n)] *= 0
            return z

        means = torch.mean(z, dim=0, keepdim=True)
        means -= torch.mean(means) / 2
        z -= means.t()
        z -= means
        return z

    @classmethod
    def kernel(cls, z):
        raise NotImplemented

    @classmethod
    def sim_measure(cls, z, h, return_components=False, indices=None, attention_mask=None):
        # As along as batch size < 90, this computation is faster than direct multiplication
        z = z.view((z.shape[0], -1))
        h = h.view((h.shape[0], -1))
        K_z = cls.center(cls.kernel(z), unbiased=True)
        K_h = cls.center(cls.kernel(h), unbiased=True)

        # if indices is not None:
        #     with torch.no_grad():
        #         x_norm = torch.norm(K_z, p='fro').detach()
        #         y_norm = torch.norm(K_h, p='fro').detach()
        #     K_z = K_z / x_norm
        #     K_h = K_h / y_norm
        #
        #     bias_bias_indices = torch.cartesian_prod(indices, indices)
        #     bias_bias_indices = (bias_bias_indices[:, 0] * K_z.shape[0]) + bias_bias_indices[:, 1]
        #     K_z = K_z.flatten()
        #     K_h = K_h.flatten()
        #     scaled_hsic = torch.sum(K_h[bias_bias_indices] * K_z[bias_bias_indices])
        #     with torch.no_grad():
        #         mask = torch.ones_like(K_h, dtype=torch.bool)
        #         mask[bias_bias_indices] = False
        #         other_exp = torch.sum(K_h[mask] * K_z[mask])
        #     return other_exp + scaled_hsic

        if indices is not None:
            product = K_h * K_z
            scaled_hsic = (product)[indices, :].sum()
            with torch.no_grad():
                from numpy import setdiff1d, arange
                non_bias_indices = setdiff1d(arange(K_h.shape[0]), indices, assume_unique=True)
                other_term = product[non_bias_indices, :].sum()
                x_norm_term = (K_z[non_bias_indices] ** 2).sum()
                y_norm_term = (K_h[non_bias_indices] ** 2).sum()

            x_norm = ((K_z[indices] ** 2).sum() + x_norm_term).sqrt()
            y_norm = ((K_h[indices] ** 2).sum() + y_norm_term).sqrt()
            scaled_hsic = scaled_hsic + other_term
        else:
            scaled_hsic = torch.sum(K_h * K_z)
            x_norm = torch.norm(K_z, p='fro')
            y_norm = torch.norm(K_h, p='fro')
        if return_components:
            return K_h * K_z, x_norm, y_norm

        return scaled_hsic / (x_norm * y_norm)

    @classmethod
    def sim_measure_combinations(cls, z, h, bias_indices):
        RSM_prod, x_norm, y_norm = cls.sim_measure(z, h, return_components=True)
        # TODO: consider subtracting the length of main diagonal from computation, since it only zeros.
        norm_factor = (x_norm * y_norm)
        # 1. sim betweeen biased an rest
        bias_rest = (RSM_prod[bias_indices, :].sum() * (RSM_prod.numel() / RSM_prod[bias_indices, :].numel())) / norm_factor
        # 2. sim between biased and biasedlink
        bias_bias_indices = torch.cartesian_prod(bias_indices, bias_indices)
        bias_bias_indices = (bias_bias_indices[:, 0] * RSM_prod.shape[0]) + bias_bias_indices[:, 1]
        bias_bias_sum = RSM_prod.flatten()[bias_bias_indices].sum()
        bias_bias = (bias_bias_sum * (RSM_prod.numel() / bias_bias_indices.numel()))/ norm_factor
        # 3. sim between rest and rest
        overall_sum = RSM_prod.sum()
        rest_rest = ((overall_sum - RSM_prod[bias_indices, :].sum()) * (RSM_prod.numel() / (RSM_prod.numel() - RSM_prod[bias_indices, :].numel()))) / norm_factor
        # 4. overall sim
        overall = overall_sum / norm_factor
        return {'overall': overall, 'rest_rest': rest_rest, 'bias_bias': bias_bias, 'bias_rest': bias_rest}


class LinCKA(CKAReg):
    @classmethod
    def kernel(cls, z):
        return z @ z.t()

