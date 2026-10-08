# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import ListConfig
import json
import random
import os
from typing import List, Union

import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils.dataset.interface_perm import (draw_view, reorder_view)
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


_CLEAN_REDRAW = os.environ.get('IDENTITY_CLEAN_REDRAW', '0').strip().lower() in ('1', 'true', 'yes')
_DISTRACTOR_REDRAW = os.environ.get('IDENTITY_DISTRACTOR_REDRAW', '1').strip().lower() in ('1', 'true', 'yes')
_VIEW_FAMILY = os.environ.get('IDENTITY_VIEW_FAMILY', 'relabel').strip().lower()
if _VIEW_FAMILY not in ('relabel', 'order', 'order+relabel'):
    raise ValueError("IDENTITY_VIEW_FAMILY must be 'relabel', 'order' or 'order+relabel', got %r"
                     % _VIEW_FAMILY)



def _fc_row(meta):
    if not isinstance(meta, dict):
        return None
    t = meta.get('tools')
    return t if isinstance(t, list) and t else None


def _draw(template, identity_of, rng, permute_args=True):
    return draw_view(template, identity_of, rng, permute_args=permute_args)


def _reorder(template, rng):
    return reorder_view(template.get('sys', ''), rng)


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 use_chat_template=True,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error'):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        
        self.use_chat_template = use_chat_template

        self._download()
        self._read_files_and_tokenize()


    def _download(self):
        from verl.utils.fs import copy_local_path_from_hdfs
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')



        print(f'filter dataset len: {len(self.dataframe)}')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)
        cp_pre = (row_dict.get('extra_info') or {}).get('cp_meta')
        cp_pre = (json.loads(cp_pre) if isinstance(cp_pre, str) else cp_pre) if cp_pre is not None else None

        if self.use_chat_template:
            prompt = self.tokenizer.apply_chat_template(chat, tools=_fc_row(cp_pre), tokenize=False,
                                                       add_generation_prompt=True)
        else:
            prompt = ""
            for p in prompt:
                prompt += p["content"].strip() + "\n\n"
            prompt = prompt.strip()
        
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt, tokenizer=self.tokenizer, max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id, left_pad=True, truncation=self.truncation
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat

        extra_info = row_dict.get("extra_info", {})
        index = extra_info.get("index", 0)
        row_dict["index"] = index
        if "cp_meta" in extra_info:
            row_dict["extra_info"] = {k: v for k, v in extra_info.items() if k != "cp_meta"}

        cp_meta_raw = extra_info.get("cp_meta", None)
        if cp_meta_raw is not None:
            cp_meta = cp_pre if cp_pre is not None else (
                json.loads(cp_meta_raw) if isinstance(cp_meta_raw, str) else cp_meta_raw)
            row_dict["cp_meta"] = cp_meta
            row_dict["cp_row_id"] = torch.tensor(index, dtype=torch.long)
            if isinstance(cp_meta, dict):
                ds = cp_meta.get("distractors") or []
                rng = random.Random(torch.randint(0, 2 ** 31 - 1, (1,)).item())
                if _DISTRACTOR_REDRAW and len(ds) > 1:
                    ds = [ds[rng.randrange(len(ds))]]
                base = dict(ds[0]) if ds else {
                    "sys": cp_meta.get("distractor_sys") or cp_meta.get("sys", ""),
                    "user": cp_meta.get("distractor_user") or cp_meta.get("user", ""),
                    "gold": "", "identity_of": cp_meta.get("identity_of")}
                grm = row_dict.get('reward_model')
                if not (base.get("gold") or "").strip():
                    base["gold"] = grm.get('ground_truth', '') if isinstance(grm, dict) else ''
                ordered = _reorder(base, rng)
                if ordered is not None:
                    base = ordered if isinstance(ordered, dict) else dict(base, sys=ordered)
                view = _draw(base, base.get("identity_of"), rng)
                cp_meta = dict(cp_meta)
                cp_meta["distractors"] = [view]
                row_dict["cp_meta"] = cp_meta
                row_dict["distractor_index"] = torch.tensor(0, dtype=torch.long)
                d_sys, d_user = view["sys"], view["user"]
                d_chat = [{"role": "system", "content": d_sys},
                          {"role": "user", "content": d_user}]
                if self.use_chat_template:
                    d_prompt = self.tokenizer.apply_chat_template(
                        d_chat, tools=view.get("tools"), tokenize=False, add_generation_prompt=True)
                else:
                    d_prompt = (d_sys.strip() + "\n\n" + d_user.strip()).strip()
                d_input_ids, d_attention_mask = verl_F.tokenize_and_postprocess_data(
                    prompt=d_prompt, tokenizer=self.tokenizer, max_length=self.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id, left_pad=True, truncation=self.truncation)
                row_dict["distractor_input_ids"] = d_input_ids[0]
                row_dict["distractor_attention_mask"] = d_attention_mask[0]
                row_dict["distractor_position_ids"] = compute_position_id_with_mask(d_attention_mask)[0]
        return row_dict
