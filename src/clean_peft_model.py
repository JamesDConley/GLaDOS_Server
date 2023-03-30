import sys
import os
import warnings
import logging

import torch
from torch import nn
from stop_criteria import StopOnStr
from peft import get_peft_model, LoraConfig, TaskType

warnings.filterwarnings("ignore")

from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig

import deepspeed

os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)

class GLaDOS:
    def __init__(self, path, stop_phrase="User :", base_model_path = "EleutherAI/gpt-neo-1.3B",  device="cuda", half=False, cache_dir="models/hface_cache", use_deepspeed=True, int8=False, max_length=2048):
        if int8:
            model = AutoModelForCausalLM.from_pretrained(base_model_path, return_dict=True, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.float16, load_in_8bit=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(base_model_path, cache_dir=cache_dir)
        self.device = device
        self.half = half
        self.base_model_path = base_model_path
        self.model_path = path
        self.stop_phrase = stop_phrase
        self.max_length = max_length
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=True, r=16, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        model.eval()
        self.model = model
        if half:
            self.model.half()
        self.model.load_state_dict(torch.load(self.model_path))
        
        
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, truncation_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.stop_token_seqs = [StopOnStr(stop_phrase, self.tokenizer)]
        bad_words = ["OpenAI", "GLaDOS :", "OpenAI's"]
        whitespace_words = ["\n", " ", "\t"]
        self.whitespace_tokens = [self.tokenizer(words, add_special_tokens=False).input_ids for words in whitespace_words]
        self.bad_token_seqs = [self.tokenizer(words, add_special_tokens=False).input_ids for words in bad_words]
        if use_deepspeed:
            self.ds_engine = deepspeed.init_inference(self.model,
                                 mp_size=1,
                                 dtype=torch.half,
                                 checkpoint=None,
                                 replace_with_kernel_inject=False)
            self.model = self.ds_engine.module

    def add_bad_phrase(self, bad_phrase):
        self.bad_token_seqs.append(StopOnStr(bad_phrase, self.tokenizer).stop_ids)

    def add_stop_phrase(self, stop_phrase):
        self.stop_token_seqs.append(StopOnStr(stop_phrase, self.tokenizer))
    
    def run_model(self, text, kwargs=None):
        base_kwargs = {
            "num_beams" : 16,
            "stopping_criteria" : self.stop_token_seqs,
            "max_new_tokens" : 250,
            "pad_token_id" : self.tokenizer.eos_token_id,
            "repetition_penalty" : 2.0,
            "bad_words_ids" : self.bad_token_seqs,
            "no_repeat_ngram_size" : 5,
            
        }
        # Update defaults
        if kwargs is not None:
            for key, item in kwargs.items():
                base_kwargs[key] = item

        
        logger.info(f"Base length : {len(self.tokenizer(text, return_tensors='pt').input_ids[0])}")
        input_ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length - base_kwargs["max_new_tokens"]).input_ids.to(self.device)
        logger.info(f"Truncated length : {len(input_ids[0])} ")
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                gen_tokens = self.model.generate(
                    input_ids=input_ids,
                    **base_kwargs
                )
        # slice out the input sequence
        gen_tokens = gen_tokens[:, input_ids.shape[-1]:]
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        #gen_text = gen_text[len(self.tokenizer.batch_decode([input_ids]))]
        return gen_text

    def converse(self, user_input, conversation_history=None, kwargs=None, truncate=True, speaker="User"):
        if conversation_history is not None:
            speakers = ["User", "GLaDOS"]
            convo_txt = ""
            for idx, message in enumerate(conversation_history):
                convo_txt+= f"{speakers[idx % 2]} :\n{message}\n"
        else:
            convo_txt = ""
        convo_txt += f"{speaker} :\n{user_input}"
        convo_txt += "\nGLaDOS :\n"
        prompt = convo_txt
        logging.info(convo_txt)
        
        new_text = self.run_model(prompt, kwargs=kwargs)
        logging.info(new_text)
        # TODO : This doesn't work because the input gets truncated inside
        if truncate:
            new_text = new_text.split("<|endoftext|>")[0]
            new_text = new_text.split(self.stop_phrase)[0]
            new_text = new_text.split("<|endoftext|>")[0]
        return new_text


        