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

import os
import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

WorkerType = Type[Worker]


class Role(Enum):
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()

    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    
    sequence_score_format = batch.batch['token_level_scores_format'].sum(-1)
    sequence_score_correctness = batch.batch['token_level_scores_correctness'].sum(-1)
    sequence_score_length = batch.batch['token_level_scores_length'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        'critic/format_score/mean':
            torch.mean(sequence_score_format).detach().item(),
        'critic/format_score/max':
            torch.max(sequence_score_format).detach().item(),
        'critic/format_score/min':
            torch.min(sequence_score_format).detach().item(),
        'critic/correctness_score/mean':
            torch.mean(sequence_score_correctness).detach().item(),
        'critic/correctness_score/max':
            torch.max(sequence_score_correctness).detach().item(),
        'critic/correctness_score/min':
            torch.min(sequence_score_correctness).detach().item(),
        'critic/length_score/mean':
            torch.mean(sequence_score_length).detach().item(),
        'critic/length_score/max':
            torch.max(sequence_score_length).detach().item(),
        'critic/length_score/min':
            torch.min(sequence_score_length).detach().item(),
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):

    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):


        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='left')
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=2,
                                           persistent_workers=True,
                                           prefetch_factor=2,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='left')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        reward_tensor_lst = []
        format_tensor_lst = []
        correctness_tensor_lst = []
        length_tensor_lst = []
        
        data_source_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor, format_tensor, correctness_tensor, length_tensor = self.val_reward_fn(test_batch, self.global_steps)

            reward_tensor_lst.append(reward_tensor)
            format_tensor_lst.append(format_tensor)
            correctness_tensor_lst.append(correctness_tensor)
            length_tensor_lst.append(length_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        format_tensor = torch.cat(format_tensor_lst, dim=0).sum(-1).cpu()
        correctness_tensor = torch.cat(correctness_tensor_lst, dim=0).sum(-1).cpu()
        length_tensor = torch.cat(length_tensor_lst, dim=0).sum(-1).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)
        
        data_source_reward = {}
        data_source_format = {}
        data_source_correctness = {}
        data_source_length = {}
        
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
                data_source_format[data_source] = []
                data_source_correctness[data_source] = []
                data_source_length[data_source] = []
            
            data_source_reward[data_source].append(reward_tensor[i].item())
            data_source_format[data_source].append(format_tensor[i].item())
            data_source_correctness[data_source].append(correctness_tensor[i].item())
            data_source_length[data_source].append(length_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
            metric_dict[f'val/test_format/{data_source}'] = np.mean(data_source_format[data_source])
            metric_dict[f'val/test_correctness/{data_source}'] = np.mean(data_source_correctness[data_source])
            metric_dict[f'val/test_length/{data_source}'] = np.mean(data_source_length[data_source])

        return metric_dict

    
    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()
        world_size = self.actor_rollout_wg.world_size
        uids = batch.non_tensor_batch.get('uid')
        groups = {}
        if uids is not None:
            for i, u in enumerate(uids):
                groups.setdefault(u, []).append(i)
        if groups and len(groups) % world_size == 0:
            keys = list(groups)
            r = batch.batch['responses'].shape[-1]
            prompt_len = attention_mask[:, :-r].sum(-1).tolist()
            resp_len = attention_mask[:, -r:].sum(-1).tolist()
            group_cost = [max(prompt_len[i] for i in groups[k]) + sum(resp_len[i] for i in groups[k])
                          for k in keys]
            group_parts = get_seqlen_balanced_partitions(group_cost, k_partitions=world_size, equal_size=True)
            global_partition_lst = [[i for g in part for i in groups[keys[g]]] for part in group_parts]
            if len({len(p) for p in global_partition_lst}) != 1:
                global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                                     k_partitions=world_size,
                                                                     equal_size=True)
        else:
            global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                                  k_partitions=world_size,
                                                                  equal_size=True)
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0


        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        self.global_steps += 1


        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                has_distractor = ('distractor_input_ids' in batch.batch.keys()
                                  and self.config.actor_rollout_ref.rollout.get('distractor_rollout', True))
                if has_distractor:
                    distr_pop = batch.pop(batch_keys=[
                        'distractor_input_ids', 'distractor_attention_mask', 'distractor_position_ids'])
                    flat = {k: (distr_pop.batch[k][:, 0] if distr_pop.batch[k].dim() == 3
                                else distr_pop.batch[k])
                            for k in ('distractor_input_ids', 'distractor_attention_mask',
                                      'distractor_position_ids')}
                    distr_gen_batch = DataProto.from_dict(tensors={
                        'input_ids': flat['distractor_input_ids'],
                        'attention_mask': flat['distractor_attention_mask'],
                        'position_ids': flat['distractor_position_ids'],
                    }, meta_info={'recompute_log_prob': bool(
                        self.config.actor_rollout_ref.actor.get('distractor_in_pg', False))})

                with _timer('step', timing_raw):
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        if has_distractor:
                            distr_gen_batch_output = self.actor_rollout_wg.generate_sequences(distr_gen_batch)

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    if has_distractor:
                        batch.batch['distractor_responses'] = distr_gen_batch_output.batch['responses']
                        batch.batch['distractor_full_ids'] = distr_gen_batch_output.batch['input_ids']
                        batch.batch['distractor_full_mask'] = distr_gen_batch_output.batch['attention_mask']
                        d = distr_gen_batch_output
                        d.non_tensor_batch['uid'] = np.array(
                            [u + '_d' for u in batch.non_tensor_batch['uid']], dtype=object)
                        rm = list(batch.non_tensor_batch['reward_model'])
                        cm = list(batch.non_tensor_batch['cp_meta'])
                        midx_all = (batch.batch['distractor_index'].tolist()
                                    if 'distractor_index' in batch.batch.keys() else [0] * len(rm))
                        d_rm, d_cm = [], []
                        for i in range(len(rm)):
                            meta = cm[i]
                            meta = json.loads(meta) if isinstance(meta, str) else dict(meta)
                            menus = meta.get('distractors') or []
                            k = int(midx_all[i])
                            if not (0 <= k < len(menus)):
                                raise RuntimeError("row %d was rolled out on view %d but carries %d "
                                                   "view(s); its reference answer cannot be resolved"
                                                   % (i, k, len(menus)))
                            ds = menus[k]
                            dm = dict(meta)
                            dm['sys'], dm['user'] = ds['sys'], ds['user']
                            dm['identity_of'] = ds['identity_of']
                            dm['pg_distractor'] = True
                            d_rm.append({'style': 'rule', 'ground_truth': ds['gold']})
                            d_cm.append(dm)
                        d.non_tensor_batch['reward_model'] = np.array(d_rm, dtype=object)
                        d.non_tensor_batch['cp_meta'] = np.array(d_cm, dtype=object)
                        d.batch['distractor_responses'] = d.batch['responses']
                        for k in list(batch.batch.keys()):
                            if k in d.batch.keys():
                                continue
                            v = batch.batch[k]
                            d.batch[k] = (v + 100000) if k == 'cp_row_id' else v.clone()
                        for k, v in batch.non_tensor_batch.items():
                            if k not in d.non_tensor_batch:
                                d.non_tensor_batch[k] = np.array(list(v), dtype=object)
                        merged = DataProto.concat([batch, d])
                        order = torch.empty(len(merged.batch), dtype=torch.long)
                        half = len(batch.batch)
                        order[0::2] = torch.arange(half)
                        order[1::2] = torch.arange(half, 2 * half)
                        merged.reorder(order)
                        batch = merged
                        if 'distractor_index' not in batch.batch.keys():
                            batch.batch['distractor_index'] = torch.zeros(
                                len(batch.batch), dtype=torch.long, device=batch.batch['responses'].device)
                        else:
                            batch.batch['distractor_index'] = \
                                batch.batch['distractor_index'].to(batch.batch['responses'].device)
                        with _timer('action_dists', timing_raw):
                            _ad_in = batch.select(
                                batch_keys=['input_ids', 'attention_mask', 'responses', 'cp_row_id',
                                            'distractor_full_ids', 'distractor_full_mask',
                                            'distractor_index'],
                                non_tensor_batch_keys=['cp_meta'],
                                meta_info_keys=list(batch.meta_info.keys()))
                            _out = self.actor_rollout_wg.compute_action_dists(_ad_in)
                        metrics.update(reduce_metrics(_out.meta_info.get('metrics', {})))
                        ad = _out.batch['action_dist']
                        an = _out.batch['action_dist_n_tools']
                        got_ids = _out.batch['action_dist_row_id']
                        want_ids = batch.batch['cp_row_id']
                        n_mis = int((got_ids != want_ids).sum())
                        if n_mis:
                            raise RuntimeError(
                                "compute_action_dists returned %d of %d rows out of order; the "
                                "clean-group target cannot be attached by row"
                                % (n_mis, want_ids.numel()))
                        uids = batch.non_tensor_batch['uid']
                        ad_read = (ad.abs().sum(-1) > 0).tolist()
                        by = {}
                        for i, u in enumerate(uids):
                            if ad_read[i]:
                                by.setdefault(u, []).append(ad[i])
                        means = {u: torch.stack(v).mean(0) for u, v in by.items()}
                        tgt = torch.zeros_like(ad)
                        for i, u in enumerate(uids):
                            if u in means:
                                tgt[i] = means[u]
                        batch.batch['clean_action_dist'] = tgt
                        batch.batch['clean_n_tools'] = an
                        dad = _out.batch['distractor_action_dist']
                        cmeta = batch.non_tensor_batch['cp_meta']

                        def _is_pg_distractor(j):
                            m = cmeta[j]
                            m = json.loads(m) if isinstance(m, str) else m
                            return bool((m or {}).get('pg_distractor'))

                        is_view = [_is_pg_distractor(i) for i in range(len(uids))]
                        dad_read = (dad.abs().sum(-1) > 0).tolist()
                        dby = {}
                        for i, u in enumerate(uids):
                            if is_view[i]:
                                continue
                            if dad_read[i]:
                                dby.setdefault(u, []).append(dad[i])
                        dmeans = {u: (torch.stack(v).mean(0), len(v)) for u, v in dby.items()}
                        dtgt = torch.zeros_like(dad)
                        dcnt = torch.zeros(dad.size(0), dtype=torch.long)
                        for i, u in enumerate(uids):
                            if u in dmeans and not is_view[i]:
                                dtgt[i], dcnt[i] = dmeans[u]
                        batch.batch['distractor_group_dist'] = dtgt
                        batch.batch['distractor_group_n'] = dcnt.to(dad.device)
                        batch.pop(batch_keys=['distractor_full_ids', 'distractor_full_mask'])

                    self._balance_batch(batch, metrics=metrics)

                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    if self.use_reference_policy:
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        reward_tensor, format_tensor, correctness_tensor, length_tensor = self.reward_fn(batch, self.global_steps)
                        batch.batch['token_level_scores'] = reward_tensor
                        batch.batch['token_level_scores_format'] = format_tensor
                        batch.batch['token_level_scores_correctness'] = correctness_tensor
                        batch.batch['token_level_scores_length'] = length_tensor

                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    _is_last = (self.global_steps + 1) >= self.total_training_steps
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps % self.config.trainer.save_freq == 0 or _is_last):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
