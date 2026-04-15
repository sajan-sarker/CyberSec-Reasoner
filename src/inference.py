import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class CyberSecReasoner:
    def __init__(self, cfg):
        self.cfg = cfg
        print("TYPE:", type(self.cfg.model_path))
        print("VALUE:", self.cfg.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path)
        self.tokenizer.padding_side = self.cfg.padding_side
        self.torch_dtype = getattr(torch, self.cfg.torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.cfg.device_map,
        )
        self.model.eval()
        # enable faster inference
        if self.cfg.use_cache:
            self.model.config.use_cache = True

    def _format_prompt(self, messages):
        """ Format the prompt as needed for the model """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate(self, messages):
        prompt = self._format_prompt(messages)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=self.cfg.padding,
            truncation=self.cfg.truncation,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                do_sample=self.cfg.do_sample,
                repetition_penalty=self.cfg.repetition_penalty
            )
        
        # extract the generated part
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()

    def format_completions(self, completion):
        """ Extract:
            1. First <think>...</think> block (reasoning)
            2. Final short description + CWE ID
        
            Returns:
                reasoning: "<think>...</think>"
                final_output: "short description ... \n\nCWE-XXX"
        """
        # Extract ONLY the first think block
        think_match = re.search(r"<think>.*?</think>", completion, re.DOTALL)
        reasoning = think_match.group(0).strip() if think_match else ""
    
        # Remove ALL think blocks (even broken ones)
        cleaned_text = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL)
    
        # Remove any leftover stray tags
        cleaned_text = re.sub(r"</?think>", "", cleaned_text)
    
        # Extract last CWE ID
        cwe_matches = re.findall(r"CWE-\d+", completion)
        cwe_id = cwe_matches[-1] if cwe_matches else ""
    
        # Remove CWE mentions from description
        cleaned_text = re.sub(r"CWE-\d+", "", cleaned_text)
    
        # Normalize whitespace
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    
        # Construct final output cleanly
        final_output = f"{cleaned_text}\n\n{cwe_id}" if cwe_id else cleaned_text
    
        return reasoning, final_output