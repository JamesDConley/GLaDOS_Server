import sys
import os
import warnings
import logging

import torch
from torch import nn
from stop_criteria import StopOnStr
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel, PeftConfig

warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig 
from transformers.deepspeed import HfDeepSpeedConfig

import deepspeed
from md_utils import fix_lines

os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)

class GLaDOS:
    def __init__(self, path, stop_phrase="User :\n",  device="cuda", half=True, cache_dir="models/hface_cache", int8=False, int4=False, max_length=2048, multi_gpu=False, token=None, better_transformer=True):
        """Load the model and tokenizer in the given mode

        Args:
            path (str): Path to the PEFT pretrained model folder
            stop_phrase (str, optional): Phrase used to stop generation. Defaults to "User :".
            device (str, optional): Pytorch device "cuda", "cpu" or similar. Defaults to "cuda".
            half (bool, optional): Whether to half the model after loading it. Most loaded models are already in fp16 (half precision). Defaults to False.
            cache_dir (str, optional): Location of the huggingface cache. (Useful for usage within docker.) Defaults to "models/hface_cache".
            use_deepspeed (bool, optional): Experimental, don't recommend. Defaults to False.
            int8 (bool, optional): BROKEN/WIP. Defaults to False.
            max_length (int, optional): The maximum number of tokens the model can handle. Defaults to 2048.
            multi_gpu (bool, optional): If true the model will utilize all available GPUs. Defaults to False.
        """
        # TODO : Make int8 work
        if multi_gpu:
            device_map = "auto"
        else:
            device_map = None
        if int8:
            # THIS IS NOT TESTED
            model = AutoModelForCausalLM.from_pretrained(path, return_dict=True, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.float16, load_in_8bit=True, use_auth_token=token)
        elif int4:
            model = AutoModelForCausalLM.from_pretrained(path,
                return_dict=True, 
                cache_dir=cache_dir,
                load_in_4bit=True,
                device_map='auto',
                
                torch_dtype=torch.bfloat16,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                ))#max_memory=max_memory,
        else:
            model = AutoModelForCausalLM.from_pretrained(path, cache_dir=cache_dir, torch_dtype=torch.float16, use_auth_token=token, device_map=device_map)
            if device_map is None:
                model.to(device)
            if better_transformer:
                logger.info("Converting model to better transformer model for speedup...")
                model = model.to_bettertransformer()
       
        # Make sure it's in eval mode
        model.eval()

        # Bookkeeping
        self.device = device
        self.model_path = path
        self.stop_phrase = stop_phrase
        self.max_length = max_length
        self.model = model
        
        # Setup tokenizer
        # TODO : FIX ME : Upload tokenizer to huggingface hub under the 20b model folder so that I can revert this back to `path`
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", truncation_side="left", use_auth_token=token, cache_dir=cache_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Ban the model from generating certain phrases
        self.stop_token_seqs = [StopOnStr(stop_phrase, self.tokenizer)]
        
        # The model isn't trained on any text that contains "OpenAI" or any similar strings, but I don't want the model poking the bear
        bad_words = ["OpenAI", "GLaDOS :", "OpenAI's"]
        whitespace_words = ["\n", " ", "\t"]
        
        # TODO : Use these or remove. Wanted to restrict usage for initial model output because old versions would enter some whitespace and exit rather than saying anything
        self.whitespace_tokens = [self.tokenizer(words, add_special_tokens=False).input_ids for words in whitespace_words]
        self.bad_token_seqs = [self.tokenizer(words, add_special_tokens=False).input_ids for words in bad_words]
        

    def add_bad_phrase(self, bad_phrase):
        """Prevent the model from using the given phrase. Note that depending on how the given text is tokenized is may still be possible for the model to generate some variation of it.

        Args:
            bad_phrase (str): Phrase you want the model not to say
        """
        self.bad_token_seqs.append(StopOnStr(bad_phrase, self.tokenizer).stop_ids)

    def add_stop_phrase(self, stop_phrase):
        """Force the model to stop generating text if it generates the given token

        Args:
            stop_phrase (str): Phrase that should stop token generation
        """
        self.stop_token_seqs.append(StopOnStr(stop_phrase, self.tokenizer))
    
    def run_model(self, text, truncate=True, kwargs=None):
        """Generate text with the model

        Args:
            text (str): Text to prompt the model with
            kwargs (dict, optional): Dictionary of keyword arguments which will be passed to model.generate. Defaults to None.

        Returns:
            str: Text generated by the model
        """
        base_kwargs = {
            "num_beams" : 16,
            "stopping_criteria" : self.stop_token_seqs,
            "max_new_tokens" : 1024,
            "pad_token_id" : self.tokenizer.eos_token_id,
            "bad_words_ids" : self.bad_token_seqs,
            "no_repeat_ngram_size" : 12,
            
        }
        # Update defaults
        if kwargs is not None:
            for key, item in kwargs.items():
                base_kwargs[key] = item

        
        logger.debug(f"Base length : {len(self.tokenizer(text, return_tensors='pt').input_ids[0])}")
        input_ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length - base_kwargs["max_new_tokens"]).input_ids.to(self.device)
        logger.debug(f"Truncated length : {len(input_ids[0])} ")
        with torch.no_grad():
            #with torch.cuda.amp.autocast():
            gen_tokens = self.model.generate(
                input_ids=input_ids,
                **base_kwargs
            )
        # slice out the input sequence
        gen_tokens = gen_tokens[:, input_ids.shape[-1]:]
        return self.decode_token_seq(gen_tokens[0])


    def build_prompt(self, user_input, conversation_history=None, speaker="User", bot="GLaDOS"):
        if conversation_history is not None:
            speakers = [speaker, bot]
            convo_txt = ""
            for idx, message in enumerate(conversation_history):
                convo_txt+= f"{speakers[idx % 2]} :\n{message}\n"
        else:
            convo_txt = ""
        convo_txt += f"{speaker} :\n{user_input}"
        convo_txt += "\nGLaDOS :\n"
        convo_txt = convo_txt.replace("<|endoftext|>", "")
        return convo_txt
    
    def converse(self, user_input, conversation_history=None, kwargs=None, truncate=True, speaker="User", bot="GLaDOS"):
        """Helper function for having a conversation with the bot

        Args:
            user_input (str): Users most recent message to the bot
            conversation_history (list[str], optional): List of str objs representing conversation so far. Defaults to None.
            kwargs (dict, optional): Dictionary of arguments that will override defaults in text generation when model.generate is called. Defaults to None.
            truncate (bool, optional): If True then extra generated text and end tokens will be stripped from the models output. Defaults to True.
            speaker (str, optional): Name of the non-bot speaker in the conversation. Defaults to "User".
            bot (str, optional)L Name of the bot speaker in the conversation. Defaults to "GLaDOS".

        Returns:
            str: The bot's response to the most recent message
        """
        prompt = self.build_prompt(user_input, conversation_history, speaker=speaker, bot=bot)
        logging.info(prompt)
        
        new_text = self.run_model(prompt, truncate=truncate, kwargs=kwargs)
        logging.info(new_text)
        return new_text
    
    def decode_token_seq(self, token_seq, truncate=True):
        # TODO : The wrapping on this is a bit of a hack
        gen_text = self.tokenizer.decode(token_seq)
        if truncate:
            if gen_text.startswith("<|endoftext|>"):
                gen_text = gen_text.split("<|endoftext|>")[1]
            if gen_text.startswith(self.stop_phrase):
                gen_text = gen_text.split(self.stop_phrase)[1]
            gen_text = gen_text.split("<|endoftext|>")[0]
            gen_text = gen_text.split(self.stop_phrase)[0]
            gen_text = gen_text.split("<|endoftext|>")[0]
        return gen_text
