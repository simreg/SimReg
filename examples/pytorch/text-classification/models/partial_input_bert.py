import logging
from transformers import BertForSequenceClassification


class PartialInputBert(BertForSequenceClassification):

    def __init__(self, *args, **kwargs):
        logging.info(f'Initializing PartialInput model {kwargs}')
        super(PartialInputBert, self).__init__(*args, **kwargs)
        if hasattr(self.config, 'hypothesis_only') and self.config.hypothesis_only:
            self.masked_sentence = 0  # mask premise
        else:
            self.masked_sentence = 1

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # remove part of the input from the attention mask
        attention_mask = attention_mask.clone()
        attention_mask[(attention_mask == 1) & (token_type_ids == self.masked_sentence)] = 0
        return super(PartialInputBert, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

