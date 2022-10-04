import inspect
import logging
from typing import Dict, List, Optional

import datasets
import torch.nn.modules.module
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, is_datasets_available
from torch.nn.functional import normalize
from transformers.pipelines.base import Dataset
from transformers.trainer_utils import AggregationStrategy, BiasSamplingStrategy
import numpy as np
import os.path

from models.lexical_bias_bert import ClarkLexicalBiasModel
from utils.bias_sampler import BiasBatchSampler
from models import BertWithLexicalBiasModel
from utils.similarity_utils import pwcca_distance_choose_best_layer_matrix, orthogonal_procrustes_distance
from torch.nn import DataParallel, KLDivLoss
from torch.nn.functional import log_softmax
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
    if isinstance(model, DataParallel):
        model = model.module
    return inspect.signature(model.forward).parameters.keys()


class BaseTrainer(Trainer):
    def __init__(self, evaluation_sets: List[EvaluationSet] = None, **kwargs):
        super(BaseTrainer, self).__init__(**kwargs)
        self.logging_mode = 'train'
        self.xe_loss_log = {'train': [], 'eval': []}
        self.evaluation_sets = evaluation_sets if evaluation_sets is not None else []
        self.param_mean_hist = {}
        self.tokens_sim_hist = {}

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
            for c, c_hist in self.param_mean_hist.items():
                if len(c_hist) == 0:
                    continue
                logs[f'{c}_norm'] = sum(c_hist) / len(c_hist)
                self.param_mean_hist[c] = []
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
        return loss


class SimRegTrainer(BaseTrainer):
    def __init__(
            self,
            model,
            weak_model=None,
            bias_indices=None,
            args: TrainingArguments = None,
            **kwargs
    ):
        super(SimRegTrainer, self).__init__(model=model, args=args, **kwargs)
        self.bias_sampler: BiasBatchSampler = None
        self.sim_hist = dict()
        self.regularized_layers = args.regularized_layers
        self.weak_model = weak_model
        self.weak_model_layers = args.weak_model_layers
        # assert args.regularized_layers is not None and len(args.regularized_layers) == len(args.weak_model_layers)
        # initialized to empty lists to ensure passing them by-reference.
        self.activations = [None] * len(self.regularized_layers)
        self.weak_activations = [None] * len(self.regularized_layers)
        self.reg_lambda = args.regularization_lambda
        if self.args.n_gpu > 1:
            logging.error('This code have not been tested on multi-GPU settings.')

        self.activations_hook_init = model is None
        if model is not None:
            for i in range(len(self.regularized_layers)):
                model.get_submodule(self.regularized_layers[i]).register_forward_hook(get_activation(self.activations, i))
            self.main_signature = set(get_model_signature(model))

        if weak_model is not None:
            for i in range(len(self.weak_model_layers)):
                weak_model.get_submodule(self.weak_model_layers[i]).register_forward_hook(get_activation(self.weak_activations, i))
            self.weak_signature = set(get_model_signature(weak_model))

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

        optimize_sim_loss = self.args.regularization_delay <= self.state.global_step
        batch_bias_indices = None

        if self.bias_sampler is None and self.args.regularize_only_biased and "idx" in inputs and self.logging_mode == 'train' and \
                self.bias_indices is not None:
            indices = inputs.pop('idx').cpu().numpy()
            _, batch_bias_indices, _ = np.intersect1d(indices, self.bias_indices, return_indices=True, assume_unique=True)
            batch_bias_indices = torch.tensor(batch_bias_indices, device=self.model.device)

        if "idx" in inputs:
            inputs.pop('idx')
        if 'id' in inputs:
            inputs.pop('id')

        if self.activations_hook_init:
            logging.info(f'activations_hook activated at {os.getpid()}')
            if isinstance(model, DataParallel):
                raise NotImplementedError('Activations hook is not supported on multi-GPU yet.')
            for i in range(len(self.regularized_layers)):
                self.model.get_submodule(self.regularized_layers[i]).register_forward_hook(get_activation(self.activations, i))
            self.main_signature = set(get_model_signature(self.model))
            self.activations_hook_init = False

        main_inputs = {k: inputs[k] for k in self.main_signature.intersection(inputs.keys())}
        outputs = model(**main_inputs)

        # calculate weak_model activations, TODO: Maybe replace it with a saved activations, or even save the similarity structures (RSM)?
        if self.weak_model is not None:
            with torch.no_grad():
                if self.args.separate_weak_tokenization:
                    weak_inputs = dict()
                    for k in inputs:
                        if k.startswith("weak_"):
                            weak_inputs[k[len("weak_"):]] = inputs[k]
                    self.weak_model(**weak_inputs)
                else:
                    self.weak_model(**inputs)

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
        elif optimize_sim_loss:
            sim_loss = self.compute_sim_loss(inputs, indices=batch_bias_indices)
        else:
            with torch.no_grad():
                sim_loss = self.compute_sim_loss(inputs, indices=batch_bias_indices).detach()

        self.log_loss(loss, sim_loss)
        if sim_loss is not None:
            if self.args.enforce_similarity:
                # maximizing similarity instead of minimizing
                sim_loss *= -1

            # TODO: replace hard-coded param
            if self.args.sim_multi_step_opt and self.is_in_train and self.logging_mode == 'train' and self.state.global_step % 4 != 0:
                loss = self.reg_lambda * sim_loss
            else:
                loss = self.args.main_task_lambda * loss + self.reg_lambda * sim_loss

        return (loss, outputs) if return_outputs else loss

    def compute_sim_loss(self, inputs, indices=None):
        # prevent computing similarity for batch of size 1.
        if len(inputs) == 1:
            return None

        sim_loss = None
        for i in range(len(self.activations)):
            weak_activations = self.weak_activations[i]
            main_activations = self.activations[i]
            main_activations, updated_indices = self.extract_relevant_activations(
                inputs['attention_mask'],
                main_activations,
                batch_bias_indices=indices,
                regularized_tokens=self.args.regularized_tokens[i],
                aggregation_strategy=self.args.token_aggregation_strategy[i]
            )
            if not (isinstance(self.weak_model, BertWithLexicalBiasModel) or isinstance(self.weak_model, ClarkLexicalBiasModel)):
                weak_activations, _ = self.extract_relevant_activations(
                    inputs['weak_attention_mask'] if self.args.separate_weak_tokenization else inputs['attention_mask'],
                    weak_activations,
                    batch_bias_indices=indices,
                    regularized_tokens=self.args.regularized_tokens[i],
                    aggregation_strategy=self.args.token_aggregation_strategy[i]
                )

            tmp_sm_ls = self.generic_sim_measure(main_activations, weak_activations, updated_indices, attention_mask=inputs.get('attention_mask', None))
            if self.logging_mode not in self.sim_hist:
                self.sim_hist[self.logging_mode] = {f"{k}_{v}": [] for k, v in zip(self.regularized_layers, self.weak_model_layers)}
            self.sim_hist[self.logging_mode][f"{self.regularized_layers[i]}_{self.weak_model_layers[i]}"].append(tmp_sm_ls.item())
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
            assert main_activations.dim() == 3
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
        sim_methods: Dict[str, SimRegTrainer] = {
            'cosine': CosineSimReg,
            'cos_cor': CorSimReg,
            'abs_cos_cor': CorSimReg,
            'linear_cka': LinCKA,
            'pwcca': PWCCAReg,
            'opd': OPDReg,
            'fo_cosine': FOCosine,
            'kl_loss': KLReg,
            'cross_entropy': CEReg,
            'cosine_self': CosSelfReg,
            'l2_self': L2SelfReg,
            'entropy_reg': AttentionEntropyReg,
            'aggregated_ce': AggCEReg
        }
        if self.args.regularization_method == 'abs_cos_cor':
            kwargs = {'abs_cor': True}
        else:
            kwargs = {}
        return sim_methods[self.args.regularization_method].sim_measure(z, h, indices=indices, attention_mask=attention_mask, **kwargs)


class CosineSimReg(SimRegTrainer):
    @classmethod
    def sim_measure(cls, z, h, return_components=False, indices=None):
        n = z.shape[0]
        z = z.view((z.shape[0], -1))
        h = h.view((h.shape[0], -1))
        z = normalize(z, p=2, dim=1)
        h = normalize(h, p=2, dim=1)
        z_RSM = z @ z.t()
        h_RSM = h @ h.t()
        raise NotImplementedError
        # TODO: fix this equation, it has something wrong in its logic.
        mid_op = z_RSM - (1-h_RSM)
        if return_components:
            return mid_op
        mid_op = mid_op.flatten()[1:].view(n-1, n+1)[:, :-1]
        return mid_op.norm()

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
        if indices is not None:
            z = z[indices]
            h = h[indices]
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
        # 2. sim between biased and biased
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


class PWCCAReg(SimRegTrainer):
    @classmethod
    def sim_measure(cls, z, h, return_components=False, indices=None, attention_mask=None):
        # h is weak activations
        # here we are projecting on main activations
        return pwcca_distance_choose_best_layer_matrix(z, h, backend='qr', use_layer_matrix='y')


class OPDReg(SimRegTrainer):
    @classmethod
    def sim_measure(cls, z, h, return_components=False, indices=None, attention_mask=None):
        return orthogonal_procrustes_distance(z, h)


class selfReg(SimRegTrainer):
    def compute_sim_loss(self, inputs, indices=None):
        sim_loss = None
        for i in range(len(self.activations)):
            main_activations = self.activations[i]
            main_activations, _ = self.extract_relevant_activations(
                inputs,
                main_activations,
                batch_bias_indices=indices,
                regularized_tokens=self.args.regularized_tokens[i],
                aggregation_strategy=self.args.token_aggregation_strategy[i]
            )
            if sim_loss is None:
                sim_loss = self.generic_sim_measure(main_activations, None, indices, attention_mask=inputs['attention_mask'])
            else:
                sim_loss = sim_loss + self.generic_sim_measure(main_activations, None, indices, attention_mask=inputs['attention_mask'])
        return sim_loss
        # labels = inputs['labels'][indices]
        # sim_loss = torch.tensor(0.0, device=labels.device)
        # count = 0
        # for i in range(self.model.config.num_labels):
        #     label_indices = labels == i
        #     n = torch.sum(label_indices)
        #     if n <= 1:
        #         continue
        #     k = self.sim_measure(main_activations[label_indices], weak_activations[label_indices])
        #     sim_loss = k if sim_loss is None else sim_loss + k
        #     count += (n * n-1) / 2
        # if count > 0:
        #     sim_loss /= count
        # return sim_loss


class CosSelfReg(selfReg):
    @classmethod
    def sim_measure(cls, z, h, return_components=False, indices=None):
        z = z - z.mean(dim=0)
        if indices is not None:
            z = z[indices]
        z = torch.nn.functional.normalize(z, dim=1)
        norms = torch.matmul(z, z.t())
        n = norms.shape[0]
        return norms.flatten()[1:].view(n-1, n+1)[:, :-1].abs().mean()


class L2SelfReg(selfReg):
    @classmethod
    def sim_measure(cls, z, h, return_components=False, indices=None, attention_mask=None):
        n = z.shape[0]
        z = z.contiguous()
        return -torch.cdist(z, z, p=2).flatten()[1:].view(n-1, n+1)[:, :-1].sum()


class AttentionEntropyReg(selfReg):
    @classmethod
    def sim_measure(cls, z, h, return_components=False, indices=None, attention_mask=None):
        if indices is not None:
            z = z[indices]
            attention_mask = attention_mask[indices]
        # this expression gets the probability vectors for non-PAD tokens in each attention head.
        z = log_softmax(z[attention_mask.unsqueeze(-1).expand(-1, -1, z.shape[1]).transpose(1, 2).bool()], -1)
        # taking negative of entropy, in order to maximize it.
        return (z.exp() * z).sum(dim=-1).mean()


# First order cosine similarity
class FOCosine(SimRegTrainer):
    @classmethod
    def sim_measure(cls, z, h, return_components=False, indices=None, attention_mask=None):
        # h is weak activations
        # here we are projecting on main activations
        if indices is not None:
            z = z[indices]
            h = h[indices]
            if len(z) == 0:
                return torch.tensor(0)

        z = z.view((z.shape[0], -1))
        h = h.view((h.shape[0], -1))
        z = normalize(z, p=2, dim=1)
        h = normalize(h, p=2, dim=1)
        return torch.norm((z * h).sum(dim=-1))


# Kullback-Leibler divergence
class KLReg(SimRegTrainer):
    # input must be attention scores!
    kl_loss = KLDivLoss(reduction='batchmean', log_target=True)

    @classmethod
    def extract_relevant_probabity_scores(cls, z, h, indices, attention_mask):
        if indices is not None:
            z = z[indices]
            h = h[indices]
            attention_mask = attention_mask[indices]
            if len(z) == 0:
                return torch.tensor(0)
        z = log_softmax(z[attention_mask.unsqueeze(-1).expand(-1, -1, z.shape[1]).transpose(1, 2).bool()], -1)
        h = log_softmax(h[attention_mask.unsqueeze(-1).expand(-1, -1, h.shape[1]).transpose(1, 2).bool()], -1)
        return z, h, attention_mask

    @classmethod
    def sim_measure(cls, z, h, return_components=False, indices=None, attention_mask=None):
        """
        :param z: main model attention scores
        :param h: weak model attention scores, same shape as main model
        :param indices: indices of biased samples
        :return: mean of KL divergence of probs.
        """
        z, h, attention_mask = cls.extract_relevant_probabity_scores(z, h, indices, attention_mask)
        return -cls.kl_loss(z, h)


# Cross entropy regularizer
class CEReg(KLReg):
    # input must be attention scores!
    @classmethod
    def sim_measure(cls, z, h, return_components=False, indices=None, attention_mask=None):
        """
        :param z: main model attention scores
        :param h: weak model attention scores, same shape as main model
        :param indices: indices of biased samples
        :return: negative mean of `symmetric cross entropy` between the attention of the two models.
        """
        z, h, attention_mask = cls.extract_relevant_probabity_scores(z, h, indices, attention_mask)
        return (z.exp() * h).sum(dim=-1).mean()


class AggCEReg(KLReg):
    @classmethod
    def sim_measure(cls, z, h, return_components=False, indices=None, attention_mask=None):
        if indices is not None:
            z = z[indices]
            h = h[indices]
            attention_mask = attention_mask[indices]
            if len(z) == 0:
                return torch.tensor(0)
        # activations shape: b_size, n_heads, n_tokens, n_tokens
        z = z.sum(dim=1)
        h = h.sum(dim=1)
        # activations shape: b_size, n_tokens, n_tokens
        z = z[attention_mask.bool()].log_softmax(-1)
        h = h[attention_mask.bool()].log_softmax(-1)
        return (z.exp() * h).sum(dim=-1).mean()
