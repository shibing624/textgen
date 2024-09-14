# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: refer https://github.com/ThilinaRajapakse/simpletransformers
"""
import copy
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers import (
    BertModel,
    BertPreTrainedModel,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    FlaubertModel,
    LongformerModel,
    XLMModel,
    XLMPreTrainedModel,
    XLNetModel,
    XLNetPreTrainedModel,
    T5ForConditionalGeneration,
)
from transformers.models.albert.modeling_albert import AlbertModel, AlbertPreTrainedModel
from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.models.longformer.modeling_longformer import LongformerClassificationHead, LongformerPreTrainedModel


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """
    Bert model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.pos_weight = pos_weight

        self.init_weights()

    def forward(
            self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class XLNetForMultiLabelSequenceClassification(XLNetPreTrainedModel):
    """
    XLNet model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(XLNetForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.d_model, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mems=None,
            perm_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
        )
        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        logits = self.logits_proj(output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class XLMForMultiLabelSequenceClassification(XLMPreTrainedModel):
    """
    XLM model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(XLMForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.transformer = XLMModel(config)
        self.sequence_summary = SequenceSummary(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            langs=None,
            token_type_ids=None,
            position_ids=None,
            lengths=None,
            cache=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
        )

        output = transformer_outputs[0]
        logits = self.sequence_summary(output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep new_mems and attention/hidden states if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class AlbertForMultiLabelSequenceClassification(AlbertPreTrainedModel):
    """
    Alber model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(AlbertForMultiLabelSequenceClassification, self).__init__(config)

        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class FlaubertForMultiLabelSequenceClassification(FlaubertModel):
    """
    Flaubert model adapted for multi-label sequence classification
    """

    def __init__(self, config, pos_weight=None):
        super(FlaubertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.transformer = FlaubertModel(config)
        self.sequence_summary = SequenceSummary(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            langs=None,
            token_type_ids=None,
            position_ids=None,
            lengths=None,
            cache=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
        )

        output = transformer_outputs[0]
        logits = self.sequence_summary(output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep new_mems and attention/hidden states if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class LongformerForMultiLabelSequenceClassification(LongformerPreTrainedModel):
    """
    Longformer model adapted for multilabel sequence classification.
    """

    def __init__(self, config, pos_weight=None):
        super(LongformerForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.pos_weight = pos_weight

        self.longformer = LongformerModel(config)
        self.classifier = LongformerClassificationHead(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            global_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
    ):
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


class ElectraPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ElectraForLanguageModelingModel(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super(ElectraForLanguageModelingModel, self).__init__(config, **kwargs)
        if "generator_config" in kwargs:
            generator_config = kwargs["generator_config"]
        else:
            generator_config = config
        self.generator_model = ElectraForMaskedLM(generator_config)
        if "discriminator_config" in kwargs:
            discriminator_config = kwargs["discriminator_config"]
        else:
            discriminator_config = config
        self.discriminator_model = ElectraForPreTraining(discriminator_config)
        self.vocab_size = generator_config.vocab_size
        if kwargs.get("tie_generator_and_discriminator_embeddings", True):
            self.tie_generator_and_discriminator_embeddings()

    def tie_generator_and_discriminator_embeddings(self):
        self.discriminator_model.set_input_embeddings(self.generator_model.get_input_embeddings())

    def forward(self, inputs, labels, attention_mask=None, token_type_ids=None):
        d_inputs = inputs.clone()

        # run masked LM.
        g_out = self.generator_model(
            inputs, labels=labels, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        # get samples from masked LM.
        sample_probs = torch.softmax(g_out[1], dim=-1, dtype=torch.float32)
        sample_probs = sample_probs.view(-1, self.vocab_size)

        sampled_tokens = torch.multinomial(sample_probs, 1).view(-1)
        sampled_tokens = sampled_tokens.view(d_inputs.shape[0], -1)

        # labels have a -100 value to mask out loss from unchanged tokens.
        mask = labels.ne(-100)

        # replace the masked out tokens of the input with the generator predictions.
        d_inputs[mask] = sampled_tokens[mask]

        # turn mask into new target labels.  1 (True) for corrupted, 0 otherwise.
        # if the prediction was correct, mark it as uncorrupted.
        correct_preds = sampled_tokens == labels
        d_labels = mask.long()
        d_labels[correct_preds] = 0

        # run token classification, predict whether each token was corrupted.
        d_out = self.discriminator_model(
            d_inputs, labels=d_labels, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        g_loss = g_out[0]
        d_loss = d_out[0]
        g_scores = g_out[1]
        d_scores = d_out[1]
        return g_loss, d_loss, g_scores, d_scores, d_labels


class CopyGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.prob_proj = nn.Linear(config.d_model * 2, 1)

    def forward(self, src, decode_output, decode_attn, memory, gen_logits):
        decode_attn = torch.mean(decode_attn, dim=1)
        batch_size, steps, seq = decode_attn.size()
        src = src.unsqueeze(1).repeat([1, steps, 1])
        # vocab
        copy_logits = torch.zeros_like(gen_logits)
        context = torch.matmul(decode_attn, memory)
        copy_logits = copy_logits.scatter_add(2, src, decode_attn)
        prob = self.prob_proj(torch.cat([context, decode_output], -1)).sigmoid()

        gen_logits = prob * gen_logits.softmax(-1)
        copy_logits = (1 - prob) * copy_logits.softmax(-1)
        final_logits = gen_logits + copy_logits
        return final_logits


class CopyT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.generator = CopyGenerator(config)

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids, model_kwargs, model_input_name=None):
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
            }
            new_kwargs = copy.deepcopy(encoder_kwargs)
            new_kwargs.pop('src')
            model_kwargs["encoder_outputs"] = encoder(input_ids, return_dict=True, **new_kwargs)
        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        res = super().prepare_inputs_for_generation(input_ids=input_ids,
                                                    past=past,
                                                    attention_mask=attention_mask,
                                                    head_mask=head_mask,
                                                    decoder_head_mask=decoder_head_mask,
                                                    cross_attn_head_mask=cross_attn_head_mask,
                                                    use_cache=use_cache,
                                                    encoder_outputs=encoder_outputs,
                                                    **kwargs)
        res['src'] = kwargs['src']
        return res

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            src=None
    ):
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  decoder_input_ids=decoder_input_ids,
                                  decoder_attention_mask=decoder_attention_mask,
                                  encoder_outputs=encoder_outputs,
                                  past_key_values=past_key_values,
                                  inputs_embeds=inputs_embeds,
                                  decoder_inputs_embeds=decoder_inputs_embeds,
                                  labels=labels,
                                  use_cache=use_cache,
                                  output_attentions=True,
                                  output_hidden_states=True,
                                  return_dict=True)

        memory = outputs.encoder_last_hidden_state
        decode_attn = outputs.cross_attentions[-1]
        decode_output = outputs.decoder_hidden_states[-1]
        gen_logits = outputs.logits
        if self.training:
            prob = self.generator(input_ids, decode_output, decode_attn, memory, gen_logits)
        else:
            if src is not None:
                prob = self.generator(src, decode_output, decode_attn, memory, gen_logits)
            else:
                prob = self.generator(input_ids, decode_output, decode_attn, memory, gen_logits)
        outputs.logits = prob
        return outputs
