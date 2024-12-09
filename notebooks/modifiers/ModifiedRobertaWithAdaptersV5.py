

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:03:45 2024

@author: 15138
"""
import torch
import torch.nn as nn
from transformers import RobertaPreTrainedModel,RobertaModel, RobertaConfig, RobertaTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

class AdapterBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        # Modify this block to have the right architecture
        super(AdapterBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
class PassThroughBlock(nn.Module):
    def __init__(self):
        # Modify this block to have the right architecture
        super(PassThroughBlock, self).__init__()

    def forward(self, x):
        return x

class ModifiedRobertaLayer(nn.Module):
    def __init__(self, base_model, idx, adapter_layers, adapter_hidden_dim=2048):
        super(ModifiedRobertaLayer, self).__init__()
        # Adapter block
        if adapter_layers[idx]:
            self.adapter1 = AdapterBlock(base_model.config.hidden_size, adapter_hidden_dim) 
            self.adapter2 = AdapterBlock(base_model.config.hidden_size, adapter_hidden_dim)
        else:
            self.adapter1 = PassThroughBlock()
            self.adapter2 = PassThroughBlock()

        # Initialize the different steps of sub operations in the layer
        # attention
        currentLayer = base_model.roberta.encoder.layer[idx]
        self.attention = currentLayer.attention
        
        
        # FeedForward layers
        self.intermediate = currentLayer.intermediate
        
        self.output = currentLayer.output

    def forward(self, x, attention_mask = None):
        # Attention block
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype= torch.float32)
            
        # Convert input tensor to float (required for attention mechanism)
        x = x.to(dtype=torch.float32)
        
        # Compute the attention layer output
        attention_output = self.attention(x, attention_mask = attention_mask)
        attention_output = attention_output[0]
        
        # Compute the feed forward stage
        attention_output = self.adapter1(attention_output)
        intermediate_output = self.intermediate(attention_output)
        output = self.output(intermediate_output, attention_output)
        output = self.adapter2(output)
        
        
        return output
    def set_requires_grad(self, requires_grad = False):
        """
        Sets the requires_grad attribute for all parameters in the layer and adapter block.
        :param requires_grad: Boolean value to set the requires_grad flag.
        """
        # Set requires_grad for all parameters in the adapter block
        for param in self.adapter1.parameters():
            param.requires_grad = True
        for param in self.adapter2.parameters():
            param.requires_grad = True
        
        # Set requires_grad for all parameters in the attention and feedforward components
        for param in self.attention.parameters():
            param.requires_grad = requires_grad
        
        for param in self.intermediate.parameters():
            param.requires_grad = requires_grad
        
        for param in self.output.parameters():
            param.requires_grad = requires_grad
        

class ModifiedRobertaForSequenceClassification(nn.Module):
    def __init__(self, base_model, num_labels=2, adapter_hidden_dim=2048, adapter_layers = [False] * 12, freeze_params = False):
        
        # Extract config from the base model
        config = base_model.config
        
        # Call parent constructor with the config
        # super(ModifiedRobertaForSequenceClassification, self).__init__(config)
        
        super(ModifiedRobertaForSequenceClassification, self).__init__()

        # Initialize the base RoBERTa model (encoder only)
        # robertaModel = RobertaModel(config)
        
        self.embeddings = base_model.roberta.embeddings
        
        # Modify each layer in RoBERTa to include the adapter block
        self.modified_layers = nn.ModuleList([ModifiedRobertaLayer(base_model, idx, adapter_layers, adapter_hidden_dim) for idx in range(base_model.config.num_hidden_layers)])

        # Final classification head
        self.classifier = base_model.classifier

    
    def set_requires_grad(self, requires_grad = False):
        """
        Sets the requires_grad attribute for all parameters in the layer and adapter block.
        :param requires_grad: Boolean value to set the requires_grad flag.
        """
        # Set requires_grad for all parameters in the adapter block
        for param in self.embeddings.parameters():
            param.requires_grad = requires_grad
            #print(param)
        
        # Keep attention and feed-forward layers trainable
        for layer in self.modified_layers:
           for param in layer.attention.parameters():
             param.requires_grad = requires_grad
             #print("ok1")
           for param in layer.intermediate.parameters():
             param.requires_grad = requires_grad
             #print("ok2")
           for param in layer.output.parameters():
             param.requires_grad = requires_grad
             #print("ok3")



    def forward(self, input_ids, attention_mask=None, labels=None):
        # Pass input through the RoBERTa embeddings layer
        x = self.embeddings(input_ids)  # Shape: (batch_size, seq_length, hidden_size)
        
        # Pass through each layer of RoBERTa (modified layers with adapter)
        for layer in self.modified_layers:
            x = layer(x, attention_mask)
        
        # Final classification layer
        logits = self.classifier(x)

        # If labels are provided, calculate the loss
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            
        
        #return loss, logits
        # Return SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
