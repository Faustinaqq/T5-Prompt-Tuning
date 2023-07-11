# from torch.nn.parameter import Parameter
from transformers import T5ForConditionalGeneration as t5
import torch.nn as nn
import random
import torch
from model.t5 import T5ForConditionalGeneration as prompt_t5


class T5FineTuningModel(nn.Module):
    def __init__(self, model_name):
        super(T5FineTuningModel, self).__init__()
        self.model = t5.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad_(False)
        for param in self.get_parameters():
            param.requires_grad_(True)

    def forward(self, input_ids, target_ids):
        outputs = self.model(
            input_ids=input_ids,
            labels=target_ids,
        )
        return outputs

    def generate(self, input_ids, max_target_length):
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_target_length
        )
        return outputs

    def get_parameters(self):
        return self.model.decoder.block[-2:].parameters()
    

class T5PromptModel(nn.Module):
    def __init__(self, model_name, tokenizer, prompt_len=100, init='random', changed=False):
        super(T5PromptModel, self).__init__()
        self.trainable_token_indices = None
        tokens = ""
        token_indices = []
        # different initial method
        if init == 'class_label':
            tokens = "True False" 
        if init != 'random' and prompt_len > 0:
            if len(tokens) > 0:
                token_indices = tokenizer.encode(tokens, add_special_tokens=False) 
            trainable_token_indices = []
            if len(token_indices) < prompt_len:
                trainable_token_indices = random.sample([i for i in range(tokenizer.vocab_size) if i not in token_indices], prompt_len - len(token_indices))
            self.trainable_token_indices = token_indices + trainable_token_indices
            self.trainable_token_indices = self.trainable_token_indices[:prompt_len]
            self.trainable_token_indices = torch.LongTensor(self.trainable_token_indices)
        self.model = prompt_t5.from_pretrained(model_name, prompt_len=prompt_len, changed_token_indices=self.trainable_token_indices if changed else None)
        
        if init == 'random':
            nn.init.uniform_(self.model.prompt.weight, -0.5, 0.5)
        elif prompt_len > 0:
            self.model.prompt.weight = nn.Parameter(self.model.shared.weight[self.trainable_token_indices], requires_grad=True)
                
        for param in self.model.parameters():
            param.requires_grad_(False)
        for param in self.get_parameters():
            param.requires_grad_(True)
    
    def forward(self, input_ids, target_ids):
        # add prompt
        prompt_ids = torch.arange(self.model.prompt.num_embeddings, device=input_ids.device).reshape(1, -1).repeat(input_ids.size(0), 1)
        input_ids = torch.cat((prompt_ids, input_ids), dim=1)
        outputs = self.model(
            input_ids=input_ids,
            labels=target_ids,
        )
        return outputs

    def generate(self, input_ids, max_target_length):
        # add prompt
        prompt_ids = torch.arange(self.model.prompt.num_embeddings, device=input_ids.device).reshape(1, -1).repeat(input_ids.size(0), 1)
        input_ids = torch.cat((prompt_ids, input_ids), dim=1)
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_target_length
        )
        return outputs

    def get_parameters(self):
        return self.model.prompt.parameters()
    