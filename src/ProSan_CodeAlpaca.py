import os
import re
import copy
import math
import random
import difflib
import logging
import warnings

import torch
import spacy
import pickle
import numpy as np

from tqdm import tqdm
from rouge_score import rouge_scorer
from spacy.matcher import Matcher
from spacy.language import Language
from nltk.corpus import wordnet as wn
from datasets import load_dataset, Dataset 
from torch.utils.data import Subset
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, RobertaForMaskedLM, AutoModelForCausalLM

warnings.filterwarnings("ignore")
device = "cuda:0"

def config_logger(log_file_name):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=f'{log_file_name}.log',
        filemode='a'
    )

class ProSan:
    def __init__(self, model, tokenizer, bert_model, bert_tokenizer, template, prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.template = template
        self.prompt = prompt
    
    def _token_importance(self, prompt, content):
        query = self.template.format(prompt=prompt, content=content)
        inputs = self.tokenizer(query, return_tensors='pt').to(device)
        label = self.tokenizer.decode(
            self.model.generate(
                inputs["input_ids"],
                max_new_tokens=100,
                do_sample=True,
                top_p=0.9,
                temperature=0.5
            )[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        query_label = "".join([query, label])
        query_label_ids = self.tokenizer(query_label, return_tensors="pt").input_ids.to(device)
        query_label_masks = self.tokenizer(query_label, return_tensors="pt").attention_mask.to(device)
        embeddings = self.model.get_input_embeddings()(query_label_ids)
        embeddings.requires_grad_()
        embeddings.retain_grad()
        query_ids = self.tokenizer(query, return_tensors="pt").input_ids.to(device)
        labels = query_label_ids.clone()
        labels[0, :len(query_ids[0])] = -100

        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=query_label_masks,
            labels=labels
        )
        outputs.loss.backward()
        grads = embeddings.grad
        token_importances = [grad.norm().item() for grad in grads[0]]
        partial_query = query_label.split("<</SYS>>")[0] + "<</SYS>>"
        partial_query_ids = self.tokenizer(partial_query, return_tensors="pt").input_ids
        query_tokens = self.tokenizer.tokenize(query)
        partial_query_tokens = self.tokenizer.tokenize(partial_query)
        content_tokens_length = len(self.tokenizer.tokenize("<</SYS>>"+content)[5:])
        content_tokens = [token.replace("▁", " ").replace("<0x0A>", "\n") for token in query_tokens[len(partial_query_tokens):len(partial_query_tokens)+content_tokens_length]]
        content_token_importances = token_importances[len(partial_query_ids[0]):len(partial_query_ids[0])+content_tokens_length]

        return content_tokens, content_token_importances

    def _token_selfinfo(self, content):
        self.model.eval()
        content = self.tokenizer.bos_token+content
        with torch.no_grad():
            content_encoding = self.tokenizer(content, add_special_tokens=False, return_tensors='pt').to(device)
            outputs = self.model(**content_encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            self_info = -torch.log(probs)
        content_ids = content_encoding['input_ids']
        content_ids_expaned = content_ids[:, 1:].unsqueeze(-1)
        content_tokens_infos = self_info[:, :-1].gather(-1, content_ids_expaned).squeeze(-1).squeeze(0).tolist()
        content_tokens_probs = probs[:, :-1].gather(-1, content_ids_expaned).squeeze(-1).squeeze(0).tolist()

        return content_tokens_infos, content_tokens_probs
    
    def _unit_importance_selfinfo_prob(self, units, tokens, token_importances, token_selfinfos, token_probs): 
        current_unit_idx = 0
        current_position = 0
        importances = [[] for _ in range(len(units))]
        selfinfos = [[] for _ in range(len(units))]
        probs = [[] for _ in range(len(units))]
        for token, importance, selfinfo, prob in zip(tokens, token_importances, token_selfinfos, token_probs):
            current_position += len(token)
            if current_position == len(units[current_unit_idx]):
                importances[current_unit_idx].append(importance)
                selfinfos[current_unit_idx].append(selfinfo)
                probs[current_unit_idx].append(prob)
                current_position = 0
                current_unit_idx += 1
            elif current_position > len(units[current_unit_idx]):
                counter_num = 1
                current_position = current_position - len(units[current_unit_idx])
                current_unit_idx += 1
                while current_position >= len(units[current_unit_idx]):
                    counter_num += 1
                    current_position = current_position - len(units[current_unit_idx])
                    current_unit_idx += 1
                    if current_unit_idx >= len(units):
                        break
                partial_importance = importance/counter_num
                partial_selfinfo = selfinfo/counter_num
                partial_prob= prob/counter_num
                for _ in range(counter_num):
                    importances[(current_unit_idx-1) - _].append(partial_importance)
                    selfinfos[(current_unit_idx-1) - _].append(partial_selfinfo)
                    probs[(current_unit_idx-1) - _].append(partial_prob)
            else:
                importances[current_unit_idx].append(importance)
                selfinfos[current_unit_idx].append(selfinfo)
                probs[current_unit_idx].append(prob)
        unit_importances = [np.mean(importance) for importance in importances]
        unit_selfinfos = [np.sum(slefinfo) for slefinfo in selfinfos]
        unit_probs = [np.mean(prob) for prob in probs]
        unit_importances = self._importance_normalize(units, unit_importances)

        return units, unit_importances, unit_selfinfos, unit_probs
    
    def _importance_normalize(self, units, importances):
        units = [unit.strip() for unit in units]
        importance_dict = {}
        for unit, importance in zip(units, importances):
            if unit in importance_dict:
                importance_dict[unit] = max(importance_dict[unit], importance)
            else:
                importance_dict[unit] = importance
        importances = [importance_dict[unit] for unit in units]

        return importances
    
    def _nums_normalize(self, nums):
        min_info = np.nanmin(nums)
        max_info = np.nanmax(nums)
        normed = (nums - min_info) / (max_info - min_info)

        normed[np.isnan(normed)] = 1.0

        return normed

    def _cal_p(self, Hp):
        if Hp >= 0:
            z = math.exp(-Hp)
            s = 1.0 / (1.0 + z)
        else:
            z = math.exp(Hp)
            s = z / (1.0 + z)
        return 0.62 * s * 100
    
    def _filter_unit(self, p, importances, tags):
        target_tags = {'NOUN', 'PROPN', 'NUM'}
        filtered_importances = [importance for importance, tag in zip(importances, tags) if tag in target_tags]
        percentile = np.nanpercentile(filtered_importances, p)
        flags = ['True' if ((tag in target_tags) and (importance < percentile)) else 'False' for importance, tag in zip(importances, tags)]

        return flags

    def _replace_num(self, shape):
        gen_num = ""
        for char in shape:
            if char == 'd':
                gen_num += str(random.randint(0, 9))
            else:
                gen_num += char
                
        return gen_num
    
    def _replace_noun(self, units, index, importance, selfinfo, k=10):
        noun = units[index]
        units[index] = re.sub(r'(\s*)(\S+)', r'\g<1>{}'.format(self.bert_tokenizer.mask_token), noun)
        masked_sentence = ''.join(units)
        units[index] = noun
        masked_sentence_ids = self.bert_tokenizer(masked_sentence, return_tensors="pt", max_length=512, truncation=True).to(device)
        if not (masked_sentence_ids.input_ids == self.bert_tokenizer.mask_token_id).any(dim=1):
            return noun
        with torch.no_grad():
            logits = self.bert_model(**masked_sentence_ids).logits
        mask_token_index = (masked_sentence_ids.input_ids == self.bert_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        top_k_tokens = torch.topk(logits[0, mask_token_index], k).indices[0].tolist()
        top_k_nouns = [self.bert_tokenizer.decode(token).strip() for token in top_k_tokens]
        final_noun = self._gen_noun(top_k_nouns, importance, selfinfo, noun)

        return final_noun
    
    def _calculate_similarity(self, word1, word2):
        syn_sets1 = wn.synsets(word1)
        syn_sets2 = wn.synsets(word2)
        max_similarity = 0
        for syn1 in syn_sets1:
            for syn2 in syn_sets2:
                similarity = syn1.path_similarity(syn2)
                if similarity is not None and similarity > max_similarity:
                    max_similarity = similarity

        return max_similarity
    
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))

        return e_x / e_x.sum()
    
    def _gen_noun(self, top_k_nouns, importance, selfinfo, noun):
        noun = noun.strip()
        if noun in top_k_nouns:
            index = top_k_nouns.index(noun)
            del top_k_nouns[index]
        if noun.lower() in top_k_nouns:
            index = top_k_nouns.index(noun.lower())
            del top_k_nouns[index]
        if len(top_k_nouns)==0:
            return noun
        similarities = [self._calculate_similarity(noun, top_k_noun) for top_k_noun in top_k_nouns]
        noun_similarities = [10*(importance-selfinfo)*similarity for similarity in similarities]
        noun_probs = self._softmax(noun_similarities)
        final_noun = random.choices(top_k_nouns, weights=noun_probs, k=1)[0]

        return final_noun

    def _check_privacy(self, units, flags, privacy_infos):
        for unit, flag in zip(units, flags):
            if flag == "True":
                for privacy_info in privacy_infos:
                    if privacy_info[1] in unit:
                        privacy_infos.remove(privacy_info)
                        break
                        
        return privacy_infos

    def sanitize(self, dataset):
        attr_num = {'username':  0,
                    'corporate':  0,
                    'password': 0,
                    'configuration':  0}
        leakge_num = {'username': 0,
                      'corporate': 0,
                      'password':0,
                      'configuration': 0}
        sanitized_content_record = []

        for index, data in tqdm(enumerate(dataset)):
            prompt = self.prompt
            units = data['unit']
            tags = data['tag']
            shapes = data['shape']
            content = data["privacy_content"]
            privacy_infos = copy.deepcopy(data['privacy_info'])
            attr_num[privacy_infos[0][0]] += 1
            attr_num[privacy_infos[1][0]] += 1
            tokens, token_importances = self._token_importance(prompt, content)
            token_selfinfos, token_probs = self._token_selfinfo(content)
            _, importances, self_infos, probs = self._unit_importance_selfinfo_prob(units, tokens, token_importances, token_selfinfos, token_probs)

            selfinfos = self._nums_normalize(self_infos)
            cur_selfinfo = np.sum([x for x in self_infos if not np.isinf(x)])
            cur_entropy = cur_selfinfo/len(data['unit'])
            importances = self._nums_normalize(importances)
            filter_p = self._cal_p(cur_entropy)
            flags = self._filter_unit(filter_p, importances, tags)
            result_privacy = self._check_privacy(units, flags, privacy_infos)
            if len(result_privacy) != 0:
                for info in result_privacy:
                    leakge_num[info[0]] += 1
            options_units = copy.deepcopy(units)
            for idx, (unit, tag, shape, flag, importance, selfinfo) in enumerate(zip(units, tags, shapes, flags, importances, selfinfos)):
                if flag == "True":
                    try:
                        if tag == "NOUN" or tag == "PROPN":
                            if not shape == "X":
                                noun = self._replace_noun(units, idx, importance, selfinfo)
                                options_units[idx] = re.sub(r'(\s*)(\S+)', r'\g<1>{}'.format(noun.capitalize() if 'X' == shape[0] else noun), unit)
                        elif tag == "NUM":
                            if 'x' not in shape and 'X' not in shape:
                                options_units[idx] = re.sub(r'(\s*)(\S+)', r'\g<1>{}'.format(self._replace_num(shape)), unit)
                            else:
                                options_units[idx] = unit
                        else:
                            continue
                    except re.error as e:
                        options_units[idx] = unit
            sanitized_content = "".join(options_units)
            sanitized_content_record.append(sanitized_content)
        with open("../results/CodeAlpaca_sanitized_content.pkl", "wb") as f:
            pickle.dump(sanitized_content_record, f)
            
        privacy_result = {}
        for key in attr_num:
            if key in leakge_num:
                if attr_num[key] != 0:
                    privacy_result[key] = "{:.2f}%".format((1 - leakge_num[key] / attr_num[key]) * 100)
        logging.info('PHR:'+ str(privacy_result))

model = AutoModelForCausalLM.from_pretrained(
    "edumunozsala/llama-2-7b-int4-python-code-20k",
    torch_dtype=torch.float16,
    device_map=None
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "edumunozsala/llama-2-7b-int4-python-code-20k"
)

bert_model = RobertaForMaskedLM.from_pretrained(
    "FacebookAI/roberta-base"
).to(device)

bert_tokenizer = AutoTokenizer.from_pretrained(
    "FacebookAI/roberta-base"
)

with open("../dataset/CodeAlpaca.pkl", 'rb') as f:
    dataset = pickle.load(f)
config_logger("../results/CodeAlpaca_privacy_result")
sanitizer = ProSan(model, tokenizer, bert_model, bert_tokenizer, "<<SYS>>\n{prompt}\n<</SYS>>{content}\nResponse:", f"Use the task below and the input given to write the response, which is a programming code that can solve the task.")
sanitizer.sanitize(dataset)