from transformers import BertPreTrainedModel, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from transformers.models.bert.modeling_bert import BertSelfOutput, BertIntermediate, BertOutput, BertEmbeddings, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput



class DependencyBertSelfAttetion(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads 

        # Regularization
        self.attention_probs_dropout_prob = nn.Dropout(config.attention_probs_dropout_prob)
        self.hidden_dropout_prob = nn.Dropout(config.hidden_dropout_prob)
    
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        # Parameters for fusing
        self.osa_dsa = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.fusion_gate = nn.Sequential(
            nn.Tanh(),
            nn.Linear(2 * config.hidden_size, 1),
            nn.Sigmoid
        )
    
    def transpose_for_scores(self, x):
        # Transpose x : [B, T, C] -> x: [B, nh, T, hs]

        B,T,C = x.size()

        return x.view(B, T, self.attention_heads, self.attention_head_size).transpose(1,2)

    def forward(self, hidden_states, dependency_matrix=None):
        # x : [B, T, C]
        # dependency_matrix : [B, T, T] -> [B, 1, T, T]
        dependency_matrix = dependency_matrix.unsqueeze(1)

        B,T,C = hidden_states.size()

        q,k,v = self.transpose_for_scores(self.query(hidden_states)), self.transpose_for_scores(self.key(hidden_states)), self.transpose_for_scores(self.value(hidden_states))  # [B, nh, T, hs]

        # Original self attention
        self_attn = q @ k.transpose(-2,-1) * 1/math.sqrt(k.size(-1))    # [B, nh, T, T]
        self_attn = F.softmax(self_attn, dim=-1)    # [B, nh, T, T]
        self_attn = self.attention_probs_dropout_prob(self_attn)    # [B, nh, T, T]
        osa = torch.matmul(self_attn, v)    # [B, nh, T, nh]
        osa = osa.transpose(1,2).contiguous().view(B, T, C) # [B,T,C]
        
        # Dependency integrated self attention
        dep_self_attn = (q @ k.transpose(-2,-1)) * dependency_matrix.unsqueeze(1) * 1/math.sqrt(k.size(-1))  # [B, nh, T, T]
        dep_self_attn = F.softmax(dep_self_attn, dim=-1)
        dsa = torch.matmul(dep_self_attn, v)
        dsa = dsa.transpose(1,2).contiguous().view(B, T, C) # [B,T,C]

        return self.fusion(osa, dsa)
    
    
    def fusion(self, osa, dsa):
        fused = torch.cat([osa, dsa], dim=-1)   # [B, T, 2*C]

        # Value for fusion gate
        g = self.fusion_gate(fused)

        # return output
        output = g * osa + (1-g) * dsa
        return output



class DependencyBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = DependencyBertSelfAttetion(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, dependency_matrix=None):
        self_output = self.self(hidden_states, dependency_matrix)
        attention_output = self.output(self_output, hidden_states)
        return attention_output
    


class DependencyBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = DependencyBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, dependency_matrix=None):
        attention_output = self.attention(hidden_states, dependency_matrix)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output



class DependencyBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([DependencyBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, dependency_matrix=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, dependency_matrix)
        return hidden_states


class DependencyBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = DependencyBertEncoder(config)
        self.pooler = BertPooler(config) if config.add_pooling_layer == True else None

        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        dependency_matrix=None,
    ):
        embedding_output = self.embeddings(input_ids=input_ids, 
                                           token_type_ids=token_type_ids)
        

        encoder_output = self.encoder(
            hidden_states=embedding_output,
            dependency_matrix=dependency_matrix,
        )

        sequence_output = encoder_output
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output
        )
    

class DependencyBertForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.bert = DependencyBertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout ) if config.classifier_dropout is not None else nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(
        self,
        input_ids,
        token_type_ids=None,
        dependency_matrix=None,
        labels=None,
        ):
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. 
            If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), 
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.bert(input_ids, token_type_ids, dependency_matrix)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Compute Loss

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.last_hidden_state,
        )

