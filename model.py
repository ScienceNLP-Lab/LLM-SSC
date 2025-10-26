import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
from transformers import AutoModelForCausalLM, LogitsProcessorList,MinLengthLogitsProcessor, StoppingCriteriaList,MaxLengthCriteria
from transformers import AutoConfig
import torch.nn.functional as F
import random
import numpy as np

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
class space_thinking_llm(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"]
        )
        # model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, device_map="auto")
        self.model = get_peft_model(model, peft_config)
        self.model.print_trainable_parameters()
        self.drop = nn.Dropout(p=config.dropout_thinking_linear)
        model_config = AutoConfig.from_pretrained(config.model_name_or_path, trust_remote_code=True)
        self.hidden_size = int(model_config.hidden_size)
        # self.space_thinking_linear = nn.Linear(self.hidden_size * config.num_generate_tokens, num_labels).to(torch.bfloat16)
        self.space_thinking_linear = nn.Linear(self.hidden_size * config.num_generate_tokens, 50)
        self.space_thinking_linear_2 = nn.Linear(50, num_labels)

        self.num_labels = num_labels
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.logits_processor = LogitsProcessorList(
            [MinLengthLogitsProcessor(config.max_length + config.num_generate_tokens, eos_token_id=model.generation_config.eos_token_id), ])
        self.stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=config.max_length + config.num_generate_tokens)])
        self.config = config
        self.weightfunction = nn.Sequential(nn.Linear(2 * self.num_labels, 1), nn.Sigmoid())

    def forward(self, **batch):
        output = self.model.greedy_search(**batch, logits_processor=self.logits_processor, stopping_criteria=self.stopping_criteria, output_logits=True, output_scores=True, return_dict_in_generate=True, output_hidden_states=True)
        aggregated_label_representation = output["hidden_states"][0][-1][:, -1, :]
        if self.config.num_generate_tokens > 1:
            for i in range(1, self.config.num_generate_tokens):
                aggregated_label_representation = torch.cat((aggregated_label_representation, output["hidden_states"][i][-1][:, -1, :]), dim=1)
        # out = self.drop(aggregated_label_representation)
        out_con = self.space_thinking_linear(aggregated_label_representation)
        out = self.relu(out_con)
        out = self.space_thinking_linear_2(out)
        out = F.softmax(out, dim=1)
        return out, out_con