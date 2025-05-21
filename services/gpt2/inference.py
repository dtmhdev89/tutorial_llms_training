import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Inference:
    """Inference from pretrained model"""

    def __init__(self, model_name) -> None:
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)

    @property
    def model(self):
        """model getter"""
        return self._model
    
    @property
    def tokenizer(self):
        """tokenizer getter"""
        return self._tokenizer
    
    def prompt(self, prompt_msg):
        """Make prompt request to the model"""

        inputs = self.tokenizer(
            prompt_msg,
            return_tensors="pt"
        ).to(self.model.device)

        encoded_output = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        decoded_ouput = self.tokenizer.decode(
            encoded_output[0],
            skip_special_tokens=True
        )

        return decoded_ouput, encoded_output
    
    def evaluate_with_perplexity(self, encoded_output):
        """Compute Perplexity"""

        labels = encoded_output[:, 1:].clone()
        inputs = encoded_output[:, :-1].clone()

        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs.logits 
            # batch_size, sequence_length_of_inputs, vocab_size
            print(f"--------logits shape: {logits.shape}")

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(
            2,
            labels.unsqueeze(-1)
        ).squeeze(-1)

        nll = -selected_log_probs.sum().item()
        num_tokens = labels.numel()
        perplexity = math.exp(nll / num_tokens)

        return perplexity
