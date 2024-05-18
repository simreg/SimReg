# import logging
# import torch
# import numpy as np
#
# from typing import Dict, Union, Any, Optional, Tuple, List
# from transformers.trainer import is_sagemaker_mp_enabled, nested_detach, autocast
# from torch import nn
#
# from models import BertWithLexicalBiasModel
# from models.lexical_bias_bert import ClarkLexicalBiasModel
# from my_trainer import SimRegTrainer
#
#
# class BaseGradRegTrainer(SimRegTrainer):
#
#     def __init__(self, weak_model, **kwargs):
#         assert weak_model is not None
#         logging.info('Using gradient regularizer.')
#         super(BaseGradRegTrainer, self).__init__(**kwargs, weak_model=weak_model)
#         self.sim_hist = dict()
#
#     def compute_loss(self, model, inputs, return_outputs=False):
#         """
#                 How the loss is computed by Trainer. By default, all models return the loss in the first element.
#
#                 Subclass and override for custom behavior.
#                 """
#         if self.label_smoother is not None and "labels" in inputs:
#             labels = inputs.pop("labels")
#         else:
#             labels = None
#
#         # bad naming, this variable indicates whether we want to
#         batch_bias_indices = None
#
#         if self.bias_sampler is None and self.args.regularize_only_biased and "idx" in inputs and self.logging_mode == 'train' and \
#                 self.bias_indices is not None:
#             indices = inputs.pop('idx').cpu().numpy()
#             _, batch_bias_indices, _ = np.intersect1d(indices, self.bias_indices, return_indices=True,
#                                                       assume_unique=True)
#             batch_bias_indices = torch.tensor(batch_bias_indices, device=self.model.device)
#
#         if "idx" in inputs:
#             inputs.pop('idx')
#         if 'id' in inputs:
#             inputs.pop('id')
#
#         main_inputs = {k: inputs[k] for k in self.main_signature.intersection(inputs.keys())}
#         outputs = model(**main_inputs)
#
#         if self.args.separate_weak_tokenization:
#             weak_inputs = {'labels': inputs['labels']}
#             for k in inputs:
#                 if k.startswith("weak_"):
#                     weak_inputs[k[len("weak_"):]] = inputs[k]
#             weak_outputs = self.weak_model(**weak_inputs)
#         else:
#             weak_outputs = self.weak_model(**inputs)
#
#         # Save past state if it exists
#         if self.args.past_index >= 0:
#             self._past = outputs[self.args.past_index]
#
#         if labels is not None:
#             loss = self.label_smoother(outputs, labels)
#         else:
#             # We don't use .loss here since the model may return tuples instead of ModelOutput.
#             loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
#         if self.bias_sampler is not None and not self.bias_sampler.bias_turn and self.logging_mode == 'train':
#             sim_loss = None
#         else:
#             sim_loss = self.compute_grad_sim_loss(inputs, outputs, weak_outputs, indices=batch_bias_indices)
#
#         self.log_loss(loss, sim_loss)
#         if sim_loss is not None:
#             if self.args.enforce_similarity:
#                 # maximizing similarity instead of minimizing
#                 sim_loss *= -1
#
#             if self.args.sim_multi_step_opt and self.is_in_train and self.logging_mode == 'train' and self.state.global_step % 4 != 0:
#                 loss = self.reg_lambda * sim_loss
#             else:
#                 loss += self.reg_lambda * sim_loss
#
#         return (loss, outputs) if return_outputs else loss
#
#     def compute_sim_loss(self, inputs, indices=None):
#         raise Exception('Wrong function!')
#
#     def compute_grad_sim_loss(self, inputs, main_outputs, weak_outputs, indices=None):
#         sim_loss = None
#
#         retain_grad_graph = self.logging_mode == 'train'
#         main_grads = torch.autograd.grad(
#             main_outputs.loss,
#             inputs=self.activations,
#             create_graph=retain_grad_graph,
#             retain_graph=retain_grad_graph
#         )
#         weak_grads = torch.autograd.grad(weak_outputs.loss, inputs=self.weak_activations)
#
#         for i in range(len(main_grads)):
#             extracted_main_grads, updated_batch_bias_indices = self.extract_relevant_activations(
#                 inputs['attention_mask'],
#                 main_grads[i],
#                 indices,
#                 regularized_tokens=self.args.regularized_tokens[i],
#                 aggregation_strategy=self.args.token_aggregation_strategy[i]
#             )
#             if not (isinstance(self.weak_model, BertWithLexicalBiasModel) or isinstance(self.weak_model,
#                                                                                         ClarkLexicalBiasModel)):
#                 extracted_weak_grads, _ = self.extract_relevant_activations(
#                     inputs['weak_attention_mask'] if self.args.separate_weak_tokenization else inputs['attention_mask'],
#                     weak_grads[i],
#                     indices,
#                     regularized_tokens=self.args.regularized_tokens[i],
#                     aggregation_strategy=self.args.token_aggregation_strategy[i]
#                 )
#             else:
#                 extracted_weak_grads = weak_grads[i]
#             tmp_sm_ls = self.generic_sim_measure(
#                 z=extracted_main_grads,
#                 h=extracted_weak_grads,
#                 indices=updated_batch_bias_indices,
#                 attention_mask=inputs['attention_mask'] if 'attention_mask' in inputs else None
#             )
#             if self.logging_mode not in self.sim_hist:
#                 self.sim_hist[self.logging_mode] = {f"{k}_{v}": [] for k, v in
#                                                     zip(self.regularized_layers, self.weak_model_layers)}
#             self.sim_hist[self.logging_mode][f"{self.regularized_layers[i]}_{self.weak_model_layers[i]}"].append(
#                 tmp_sm_ls.item())
#
#             if sim_loss is None:
#                 sim_loss = tmp_sm_ls
#             else:
#                 sim_loss = sim_loss + tmp_sm_ls
#         return sim_loss
#
#     def prediction_step(
#             self,
#             model: nn.Module,
#             inputs: Dict[str, Union[torch.Tensor, Any]],
#             prediction_loss_only: bool,
#             ignore_keys: Optional[List[str]] = None,
#     ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
#         has_labels = all(inputs.get(k) is not None for k in self.label_names)
#         inputs = self._prepare_inputs(inputs)
#         if ignore_keys is None:
#             if hasattr(self.model, "config"):
#                 ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
#             else:
#                 ignore_keys = []
#
#         # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
#         if has_labels:
#             labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
#             if len(labels) == 1:
#                 labels = labels[0]
#         else:
#             labels = None
#
#         if is_sagemaker_mp_enabled():
#             from transformers.trainer import smp_forward_only, smp_nested_concat
#             raw_outputs = smp_forward_only(model, inputs)
#             if has_labels:
#                 if isinstance(raw_outputs, dict):
#                     loss_mb = raw_outputs["loss"]
#                     logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
#                 else:
#                     loss_mb = raw_outputs[0]
#                     logits_mb = raw_outputs[1:]
#
#                 loss = loss_mb.reduce_mean().detach().cpu()
#                 logits = smp_nested_concat(logits_mb)
#             else:
#                 loss = None
#                 if isinstance(raw_outputs, dict):
#                     logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
#                 else:
#                     logits_mb = raw_outputs
#                 logits = smp_nested_concat(logits_mb)
#         else:
#             if has_labels:
#                 if self.use_amp:
#                     with autocast():
#                         loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
#                 else:
#                     loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
#                 loss = loss.mean().detach()
#                 if isinstance(outputs, dict):
#                     logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
#                 else:
#                     logits = outputs[1:]
#             else:
#                 loss = None
#                 if self.use_amp:
#                     with autocast():
#                         outputs = model(**inputs)
#                 else:
#                     outputs = model(**inputs)
#                 if isinstance(outputs, dict):
#                     logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
#                 else:
#                     logits = outputs
#                 # TODO: this needs to be fixed and made cleaner later.
#                 if self.args.past_index >= 0:
#                     self._past = outputs[self.args.past_index - 1]
#
#         if prediction_loss_only:
#             return (loss, None, None)
#
#         logits = nested_detach(logits)
#         if len(logits) == 1:
#             logits = logits[0]
#
#         return (loss, logits, labels)
