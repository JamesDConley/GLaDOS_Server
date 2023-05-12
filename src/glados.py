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

from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig

from transformers import GPTNeoXConfig, GPTNeoXModel

import deepspeed
from md_utils import fix_lines


os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)

class GLaDOS:
    def __init__(self, path, stop_phrase="User :\n",  device="cuda", half=False, cache_dir="models/hface_cache", use_deepspeed=False, int8=False, max_length=2048, multi_gpu=False, token=None, better_transformer=False):
        """AI is creating summary for __init__

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
        config = PeftConfig.from_pretrained(path)
        base_model_path = config.base_model_name_or_path
        
        # TODO : Make int8 work
        if int8:
            # THIS IS NOT TESTED
            model = AutoModelForCausalLM.from_pretrained(base_model_path, return_dict=True, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.float16, load_in_8bit=True, use_auth_token=token)
            # Less than half!
            device = None
            model = PeftModel.from_pretrained(model, path, return_dict=True, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.float16, load_in_8bit=True, use_auth_token=token)
        
        # TODO : Make multi_gpu work (It used to work, when did it break?)
        elif multi_gpu:
            model = AutoModelForCausalLM.from_pretrained(base_model_path, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.float16, use_auth_token=token)
            # Model should already be half
            half=True
            # Device map will be set automatically above, setting another device map break it
            model = PeftModel.from_pretrained(model, path, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.float16, use_auth_token=token)
        else:
            # TODO : Create custom device map to load on single GPU without using intermediate 
            model = AutoModelForCausalLM.from_pretrained(base_model_path, cache_dir=cache_dir, torch_dtype=torch.float16, use_auth_token=token)
            if better_transformer:
                logger.info("Converting model to better transformer model for speedup...")
                model = model.to_bettertransformer()
            model = PeftModel.from_pretrained(model, path, cache_dir=cache_dir)
            # TODO : Does this do anything? Model should already be fp16. Would be nice to remove another argument from the long list
            if half:
                model.half()
                logger.debug("Halved Model")
            if device is not None:
                model.to(device)
        
       
        # Make sure it's in eval mode
        model.eval()
        
        

        # Bookkeeping
        self.device = device
        self.base_model_path = base_model_path
        self.model_path = path
        self.stop_phrase = stop_phrase
        self.max_length = max_length
        self.model = model
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, truncation_side="left", use_auth_token=token, cache_dir=cache_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Ban the model from generating certain phrases
        self.stop_token_seqs = [StopOnStr(stop_phrase, self.tokenizer)]
        
        # The model isn't trained on any text that contains "OpenAI" or any similar strings, but I don't want the model poking the bear
        bad_words = ["OpenAI", "GLaDOS :", "OpenAI's"]
        whitespace_words = ["\n", " ", "\t"]
        
        # TODO : Use these or remove. Wanted to restrict usage for initial model output because old versions would enter some whitespace and exit rather than saying anything
        self.whitespace_tokens = [self.tokenizer(words, add_special_tokens=False).input_ids for words in whitespace_words]
        self.bad_token_seqs = [self.tokenizer(words, add_special_tokens=False).input_ids for words in bad_words]
        
        # TODO : Haven't tested this for several versions. It is probably broken.
        # Note that even when it was working it was noticeably slower than normal generation.
        if use_deepspeed:
            self.ds_engine = deepspeed.init_inference(self.model,
                                 mp_size=1,
                                 dtype=torch.half,
                                 checkpoint=None,
                                 replace_with_kernel_inject=False)
            self.model = self.ds_engine.module

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
    
    def run_model(self, text, kwargs=None):
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
            with torch.cuda.amp.autocast():
                gen_tokens = self.model.generate(
                    input_ids=input_ids,
                    **base_kwargs
                )
        # slice out the input sequence
        gen_tokens = gen_tokens[:, input_ids.shape[-1]:]
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        return gen_text

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
        if conversation_history is not None:
            speakers = [speaker, bot]
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
        # TODO : Is the above TODO outdated?
        if truncate:
            new_text = new_text.split("<|endoftext|>")[0]
            new_text = new_text.split(self.stop_phrase)[0]
            new_text = new_text.split("<|endoftext|>")[0]
        return new_text


        