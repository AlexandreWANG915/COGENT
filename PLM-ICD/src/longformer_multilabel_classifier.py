# coding=utf-8
"""
Longformer model for multi-label ICD classification.
Supports both LAAT (Label Attention) and CLS pooling modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from transformers import LongformerModel
from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class LongformerForMultilabelClassification(LongformerPreTrainedModel):
    """
    Longformer model for multi-label classification with two modes:
    - LAAT: Label Attention mechanism (each label attends to different tokens)
    - CLS: Simple [CLS] token pooling
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.use_laat = getattr(config, 'use_laat', True)

        # Longformer encoder
        self.longformer = LongformerModel(config, add_pooling_layer=False)

        if self.use_laat:
            # LAAT: Label Attention mechanism
            # Each label learns to attend to different parts of the text
            self.first_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.second_linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
            self.third_linear = nn.Linear(config.hidden_size, config.num_labels)
        else:
            # CLS: Simple classification head
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_laat_attention: bool = False,
    ):
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len) - Token IDs
            attention_mask: (batch_size, seq_len) - Attention mask
            global_attention_mask: (batch_size, seq_len) - Global attention mask for Longformer
            labels: (batch_size, num_labels) - Multi-hot labels for BCE loss
            return_laat_attention: If True, return LAAT attention weights (only works in LAAT mode)

        Returns:
            If return_laat_attention=False: SequenceClassifierOutput with loss and logits
            If return_laat_attention=True: Tuple of (SequenceClassifierOutput, att_weights)
                where att_weights has shape (batch_size, num_labels, seq_len)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Set global attention on [CLS] token if not provided
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1  # [CLS] token

        # Longformer encoding
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # (batch_size, seq_len, hidden_size)

        laat_attention = None
        if self.use_laat:
            # LAAT: Label Attention
            # Step 1: Transform hidden states
            weights = torch.tanh(self.first_linear(hidden_states))
            # (batch_size, seq_len, hidden_size)

            # Step 2: Compute attention weights for each label
            att_weights = self.second_linear(weights)
            # (batch_size, seq_len, num_labels)

            # Step 3: Softmax over sequence length, then transpose
            att_weights = F.softmax(att_weights, dim=1).transpose(1, 2)
            # (batch_size, num_labels, seq_len)

            # Store attention weights for optional return
            laat_attention = att_weights

            # Step 4: Weighted sum of hidden states for each label
            weighted_output = att_weights @ hidden_states
            # (batch_size, num_labels, hidden_size)

            # Step 5: Final prediction
            logits = self.third_linear.weight.mul(weighted_output).sum(dim=2).add(self.third_linear.bias)
            # (batch_size, num_labels)
        else:
            # CLS: Use [CLS] token representation
            cls_output = hidden_states[:, 0, :]  # (batch_size, hidden_size)
            cls_output = self.dropout(cls_output)
            logits = self.classifier(cls_output)  # (batch_size, num_labels)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        if not return_dict:
            output = (logits,) + outputs[2:]
            result = ((loss,) + output) if loss is not None else output
            if return_laat_attention and laat_attention is not None:
                return result, laat_attention
            return result

        result = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        if return_laat_attention and laat_attention is not None:
            return result, laat_attention
        return result
