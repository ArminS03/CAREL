import warnings
from typing import Any, Dict, Optional, Type, Union
import math
import numpy as np
import torch as th
import os
import random
import zipfile
import glob
import json
import torch.cuda
from gym import spaces
from torch.nn import functional as F
import time
from utils import get_linear_burn_in_fn, get_exp_decay, RolloutBuffer
from variables import procgen_envs
from model import SHELM
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, constant_fn
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean, get_latest_run_id, get_linear_fn
from stable_baselines3.common.running_mean_std import RunningMeanStd


class SHELMPPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        lr_decay: str = "none",
        ent_decay: str = "none",
        ent_decay_factor: float = 0.9,
        n_envs: int = 1,
        start_fraction: float = 0,
        end_fraction: float = 1.0,
        min_lr: float = 1e-7,
        min_ent_coef: float = 1e-4,
        config: dict = None,
        clip_decay: int = None,
        adv_norm: bool = False,
        save_ckpt: bool = True,
        use_aux: bool = False,
        threshold: float = 2,
        apply_instruction_tracking: bool = False,
    ):

        super(SHELMPPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        self.use_aux = use_aux
        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a multiple of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )

        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        self.highest_return = 0.0
        self.config = config
        self._last_mems = None
        self.threshold = threshold
        self.apply_instruction_tracking = apply_instruction_tracking

        if lr_decay == 'none':
            self.learning_rate = constant_fn(learning_rate)
        elif lr_decay == 'linear':
            self.learning_rate = get_linear_fn(learning_rate, min_lr, end_fraction=end_fraction)
        elif lr_decay == 'linear_burn_in':
            self.learning_rate = get_linear_burn_in_fn(learning_rate, min_lr, start_fraction=start_fraction, end_fraction=end_fraction)
        elif lr_decay == 'exp':
            self.learning_rate = get_exp_decay(learning_rate, ent_decay_factor, start_fraction=start_fraction)
        else:
            raise NotImplementedError(f"Learning Rate Decay {lr_decay} not implemented")

        if ent_decay == 'none':
            self.ent_coef = constant_fn(ent_coef)
        elif ent_decay == 'linear':
            self.ent_coef = get_linear_fn(ent_coef, min_ent_coef, end_fraction=end_fraction)
        elif ent_decay == 'linear_burn_in':
            self.ent_coef = get_linear_burn_in_fn(ent_coef, min_ent_coef, start_fraction=start_fraction, end_fraction=end_fraction)
        elif ent_decay == 'exp':
            self.ent_coef = get_exp_decay(ent_coef, ent_decay_factor, start_fraction)
        else:
            raise NotImplementedError(f"Entropy Decay {ent_decay} not implemented")

        if clip_decay == 'none':
            self.clip_range = constant_fn(clip_range)
        elif clip_decay == 'linear':
            self.clip_range = get_linear_fn(clip_range, end=0, end_fraction=1)
        else:
            raise NotImplementedError(f"Clipping factor decay {clip_decay} not implemented")

        self.entropy_coef = ent_coef
        self.lr_decay = lr_decay
        self.ent_decay = ent_decay
        self.counter = 0
        self.n_envs = n_envs
        self.adv_norm = adv_norm
        self.save_ckpt = save_ckpt
        if self.adv_norm:
            self._adv_rms = RunningMeanStd(shape=())

        if _init_setup_model:
            self._setup_model()

        if seed is not None:
            self._set_seed(seed)

        self.rollout_buffer = RolloutBuffer(self.n_steps, self.observation_space['image'], self.action_space, device,
                                            gamma=gamma, gae_lambda=gae_lambda, n_envs=n_envs)

        self.policy = SHELM(env.action_space, self.observation_space['image'].shape, self.config['optimizer'],
                            self.config['learning_rate'], self.config['env'], self.config['topk'],
                            device=self.device).to(self.device)

        self.video_attn_model = VideoEmbeddingModel(embed_dim=self.rollout_buffer.hidden_dim, num_heads=4,
                                                    max_sequence_length=self.rollout_buffer.buffer_size, device=self.device).to(self.device)

    def _set_seed(self, seed: int) -> None:
        """
        Seed the different random generators.

        :param seed:
        """
        # Seed python RNG
        random.seed(seed)
        # Seed numpy RNG
        np.random.seed(seed)
        # seed the RNG for all devices (both CPU and CUDA)
        th.manual_seed(seed)
        # Deterministic operations for CuDNN, it may impact performances
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
        self.action_space.seed(seed)
        if self.env is not None and self.config['env'] not in procgen_envs:
            # procgen environments do not support setting the seed that way
            self.env.seed(seed)

    def _setup_model(self) -> None:
        super(SHELMPPO, self)._setup_model()

        # Initialize schedules for policy/value clipping
        # self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        self.ent_coef_schedule = get_schedule_fn(self.ent_coef)
        self.clip_range_schedule = get_schedule_fn(self.clip_range)

    def _update_entropy_coef(self) -> None:
        # Log the current entropy coefficient
        if self.ent_decay == 'exp':
            self.ent_coef = self.ent_coef_schedule(self._current_progress_remaining, self.counter)
        else:
            self.ent_coef = self.ent_coef_schedule(self._current_progress_remaining)
        self.logger.record("train/ent_coef", self.ent_coef)

    def _dump_sources(self, outpath) -> None:
        zipf = zipfile.ZipFile(os.path.join(outpath, 'LMRL.zip'), 'w', zipfile.ZIP_DEFLATED)
        src_files = glob.glob(f'{os.path.abspath(".")}/**/*.py', recursive=True)
        for file in src_files:
            zipf.write(file, os.path.relpath(file, '../'))

    def _dump_config(self, outpath) -> None:
        with open(os.path.join(outpath, 'config.json'), 'w') as f:
            f.write(json.dumps(self.config, indent=4, sort_keys=True))


    def _auxiliary_loss(self, video_matrix, text_matrix, video_global_embeddings, text_global_embeddings, aux_coef=0.01):
        num_samples = video_matrix.shape[0]
        similarity_mx = th.zeros((num_samples, num_samples))
        for i in range(num_samples):
                    for j in range(num_samples):
                        frame_token_similarity = self.Attention_Over_Similarity_Matrix(
                            th.matmul(video_matrix[i], text_matrix[j].T)
                        )
                        text_frame_similarity = self.Attention_Over_Similarity_Vector(
                            th.matmul(video_matrix[i], text_global_embeddings[j])
                        )
                        video_token_similarity = self.Attention_Over_Similarity_Vector(
                            th.matmul(text_matrix[j], video_global_embeddings[i]).T
                        )
                        video_text_similarity = th.matmul(
                            video_global_embeddings[i].T, text_global_embeddings[j]
                        )
                        similarity_mx[i][j] = (
                            frame_token_similarity
                            + text_frame_similarity
                            + video_token_similarity
                            + video_text_similarity
                        ) / 4
        aux_loss = self.calculate_contrastive_loss(similarity_mx) * aux_coef
        return aux_loss

    def calculate_contrastive_loss(self, similarity_matrix):
        v_2_t_loss = 0
        t_2_v_loss = 0
        transposed_similarity_matrix = similarity_matrix.T
        log_softmax_row_wise = F.log_softmax(similarity_matrix, dim=1)
        log_softmax_column_wise = F.log_softmax(transposed_similarity_matrix, dim=1)

        v_2_t_loss = th.trace(log_softmax_row_wise)

        v_2_t_loss_new = v_2_t_loss / -log_softmax_row_wise.shape[0]

        t_2_v_loss = th.trace(log_softmax_column_wise)

        t_2_v_loss_new = t_2_v_loss / -log_softmax_column_wise.shape[0]

        total_loss = v_2_t_loss_new + t_2_v_loss_new

        return total_loss

    def Attention_Over_Similarity_Vector(self, vector, temp=1):
        vector_tmp = vector / temp
        attn_weights = F.softmax(vector_tmp, dim=0)
        weighted_sum = th.dot(attn_weights, vector)
        return weighted_sum

    def Attention_Over_Similarity_Matrix(self, matrix, temp=1):
        matrix_tmp = matrix / temp
        attn_col_weights = F.softmax(matrix_tmp, dim=0)
        col_product = th.mul(attn_col_weights, matrix)
        col_sum = th.sum(col_product, dim=0)
        weighted_col_sum = self.Attention_Over_Similarity_Vector(col_sum, temp)

        attn_row_weights = F.softmax(matrix_tmp, dim=1)
        row_product = th.mul(attn_row_weights, matrix)
        row_sum = th.sum(row_product, dim=1).reshape(-1)
        weighted_row_sum = self.Attention_Over_Similarity_Vector(row_sum, temp)

        return (weighted_col_sum + weighted_row_sum) / 2

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        self._update_entropy_coef()
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        else:
            clip_range_vf = None

        self.policy.train()
        # Set TrXL to evaluation mode,
        self.policy.model.eval()
        
        ################
        ### aux loss ###
        ################
        
        print(f'use aux: {self.use_aux}')
        if self.use_aux:
            frame_embeds, instr_embeds, returns = self.aux_buffer.get()
            returns = np.sum(returns, axis=1)
            indices = np.array(list(np.where(returns > 0)[0]))

            auxiliary_loss = 0
            if indices.shape[0] > 1:
                video_matrix = frame_embeds[indices]
                video_matrix = self.policy.up_project(video_matrix)
                instrs_obs = instr_embeds[indices]
                text_matrix = self.policy.get_instr_embeddings(instrs_obs)
                text_global_matrix = th.mean(text_matrix, dim=1)
                video_matrix = self.video_attn_model(video_matrix)
                video_global_matrix = th.mean(video_matrix, dim=1)

                # if self._n_updates <= 300:
                #     aux_coef = 0.1
                # else:
                #     aux_coef = 0.1 * (1 - math.log(self._n_updates - 300, 2000))
                aux_coef = 0.1
                    
                print(f'n_updates: {self._n_updates}')    
                print(f'aux coef: {aux_coef}')

                aux_loss = self._auxiliary_loss(video_matrix, text_matrix,
                    video_global_matrix, text_global_matrix, aux_coef=aux_coef)

                # Optimization step
                self.policy.optimizer.zero_grad(set_to_none=True)
                if hasattr(self.policy, 'trxl_optimizer'):
                    self.policy.trxl_optimizer.zero_grad(set_to_none=True)
                aux_loss.backward(retain_graph=True)
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                auxiliary_loss = aux_loss.detach().cpu().numpy()
        
        
        ###################################################################################################

        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses = [], []
        clip_fractions = []
        exp_vars = []
        approx_kl_divs = []
        losses = []
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):

            generator = self.rollout_buffer.get(n_batches=self.config['n_batches'])

            for rollout_data, instrs in generator:

                hiddens = rollout_data.hiddens
                actions = rollout_data.actions
                advantages = rollout_data.advantages
                old_values = rollout_data.old_values
                observations = rollout_data.observations
                old_log_prob = rollout_data.old_log_prob
                returns = rollout_data.returns
                instrs = instrs.reshape(-1)

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = actions.long()

                values, log_prob, entropy = self.policy.evaluate_actions(hiddens, actions, observations, instrs)

                # Normalize advantage
                if len(advantages) > 1:
                    values = values.squeeze()
                    self.logger.record('train/adv_std', advantages.std().detach().cpu().item())
                    self.logger.record('train/adv_mean', advantages.mean().detach().cpu().item())
                    if self.adv_norm:
                        # use running stats for advantage normalization
                        self._adv_rms.update(advantages.detach().cpu().numpy())
                        advantages = self._norm_advs(advantages)
                    else:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - old_log_prob)

                if isinstance(self.action_space, spaces.Box):
                    advantages = advantages.unsqueeze(-1)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                # Logging

                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = old_values + th.clamp(values - old_values, -clip_range_vf, clip_range_vf)
                # Value loss using the TD(gae_lambda) target
                # Normalize returns
                value_loss = F.mse_loss(returns, values_pred)

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                entropy_losses.append(entropy_loss.item())
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                clip_fractions.append(clip_fraction)

                # Re-sample the noise matrix because the log_std has changed
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # Optimization step
                self.policy.optimizer.zero_grad(set_to_none=True)
                if hasattr(self.policy, 'trxl_optimizer'):
                    self.policy.trxl_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                approx_kl_divs.append(th.mean(old_log_prob - log_prob).detach().cpu().numpy())

                exp_var = explained_variance(values.cpu().detach().numpy().flatten(),
                                             returns.cpu().detach().numpy().flatten())
                exp_vars.append(exp_var)
                losses.append(loss.detach().cpu().item())

            all_kl_divs.append(np.mean(approx_kl_divs))

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

            self._n_updates += 1

        explained_var = np.mean(exp_vars)

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", np.mean(losses))
        if self.use_aux:
            self.logger.record("train/aux_loss", auxiliary_loss)
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _norm_advs(self, advs: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        advs = advs / np.sqrt(self._adv_rms.var + epsilon)
        return advs

    def collect_rollouts(self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer,
                         n_rollout_steps: int) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        subtask_segments_scores = np.zeros((self.n_envs, 5, n_rollout_steps))
        # video_begining = np.zeros((self.n_envs), dtype=np.int64)
        instr_mask = np.zeros((self.n_envs, 5, 2))
        self.aux_buffer = DataBuffer(self.n_envs, self.device, max_envs=3)
        
        rollout_buffer.reset()
        self.policy.train()
        self.policy.model.eval()

        callback.on_rollout_start()
        # Initialize memory on first rollout collection
        if self._last_mems is None:
            self._last_mems = [torch.zeros((self.policy.mem_len, self.n_envs, self.policy.model.d_embed)).to(self.device)
                               for _ in range(self.policy.model.n_layer)]


        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                image_obs = self._last_obs['image']
                high = env.observation_space['image'].high.reshape(-1)[0]
                observations = torch.tensor(image_obs / high).float().to(self.device)
                self.policy.memory = self._last_mems
                action, value, log_prob, hidden = self.policy(observations, self._last_obs['mission'])
                self._last_mems = self.policy.memory
             
            new_obs, rewards, dones, infos = env.step(action)

            # split task to subtasks
            if self.apply_instruction_tracking:
                observation_instr = [0] * self.n_envs
                for agent_id in range(self.n_envs):
                    instr_tokens = new_obs['mission'][agent_id].replace(",", "").split(' ')
                    instr_tokens = [idx for idx, x in enumerate(instr_tokens) if x in ['and', 'then', 'after', 'before', 'or']]
                    observation_instr[agent_id] = instr_tokens

            # reset masks if done
            if self.apply_instruction_tracking:
                for proc in range(self.n_envs):
                    if dones[proc]:
                        instr_mask[proc] = np.zeros((5, 2))

            if self.apply_instruction_tracking and n_steps > 0:
                # get current active trajectory for each environment
                frame_embeds, instr_embeds = self.aux_buffer.get_first_frames(n_steps-1)
                
                # extract embedding matrices
                with torch.no_grad():
                    video_matrix = frame_embeds
                    video_matrix = self.policy.up_project(video_matrix)
                    instrs_obs = instr_embeds
                    text_matrix = self.policy.get_instr_embeddings(instrs_obs)
                    text_global_matrix = th.mean(text_matrix, dim=1)
                    video_matrix = self.video_attn_model(video_matrix)
                    video_global_matrix = th.mean(video_matrix, dim=1)

                    for proc in range(self.n_envs):
                        # instr token to whole video matrix
                        token_video_score = torch.matmul(text_matrix[proc], video_global_matrix[proc]).T
                        split_list = observation_instr[proc]
                        if not split_list:
                            continue
                        split_list.insert(0, -1)
                        for idx in range(len(split_list)): 
                            # calculate start and end of each subtask
                            start_index = split_list[idx] + 1   
                            if idx == len(split_list)-1:
                                end_index = len(new_obs['mission'][agent_id].replace(",", "").split(' '))
                            else:
                                end_index = split_list[idx+1]
                            
                            segment_score1 = np.mean(token_video_score.cpu().detach().numpy()[start_index:end_index])
                            # subtask_embedding = torch.tensor(self.preprocess_obss.instr_preproc([{'mission': 
                            #     ' '.join(self.obs[proc]['mission'].replace(",", "").split(' ')[start_index:end_index])}]), device=self.device)
                            # subtask_global_embedding = self.acmodel._get_instr_embedding(subtask_embedding)[0][0]
                            # segment_score2 = torch.matmul(video_global_embeddings[proc].T, subtask_global_embedding) # global video and global instruction score candidate
                            # semantic_score3 = numpy.max(torch.matmul(video_matrix[proc], subtask_global_embedding).cpu().detach().numpy()) # frame and subtask score candidate
                            segment_score = segment_score1
                            # append score for that subtask
                            subtask_segments_scores[proc, idx, n_steps] = segment_score
                            
                            # check if masking needed
                            if segment_score > self.threshold * np.mean(subtask_segments_scores[proc, idx, :n_steps-1]):
                                # mask with a probabilty
                                prob = 0.8 * np.tanh(self.counter/1e7) + 0.01
                                if prob >= np.random.uniform(0, 1):
                                    instr_mask[proc][idx][0], instr_mask[proc][idx][1] = start_index, end_index


                #### Applying Mask ####
                # only apply masking after forth frame
                if n_steps >= 4:
                    for proc in range(self.n_envs):
                        for seg in range(5):
                            if instr_mask[proc][seg][1] != 0:
                                if self.post_process(seg, instr_mask[proc], self.obs[proc]['mission']):
                                    instr = new_obs['mission'][agent_id].replace(",", "").split(' ')
                                    for l in range(int(instr_mask[proc][seg][0]), int(instr_mask[proc][seg][1])):
                                        instr[l] = '<mask>'
                                    instr = ' '.join(instr)
                                    new_obs['mission'][agent_id] = instr
                                else:
                                    instr_mask[proc][seg][0], instr_mask[proc][seg][1] = 0, 0  


             
            if self.use_aux and (not self.aux_buffer.is_full()):   
                image_obs = self._last_obs['image']    
                instr_obs = self._last_obs['mission']
                observations_grad = torch.tensor(image_obs / high).float().to(self.device)
                clip_embeddings = self.policy.clip_forward(observations_grad, action)
                self.aux_buffer.insert(clip_embeddings, instr_obs, rewards, dones)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            n_steps += 1
            self._update_info_buffer(infos)

            add_obs = observations.cpu().numpy()
            action = np.expand_dims(action, axis=-1)
            instr = self._last_obs['mission']
            rollout_buffer.add(add_obs, hidden, action, rewards, self._last_episode_starts, value, log_prob, instr)
            self._last_obs = new_obs
            self._last_episode_starts = dones

            for l in range(len(self._last_mems)):
                self._last_mems[l][:, self._last_episode_starts] = 0.

        with th.no_grad():
            image_obs = self._last_obs['image']
            high = env.observation_space['image'].high.reshape(-1)[0]
            observations = torch.tensor(image_obs / high).float().to(self.device)
            self.policy.memory = self._last_mems
            action, value, log_prob, hidden = self.policy(observations, self._last_obs['mission'])

        rollout_buffer.compute_returns_and_advantage(last_values=value, dones=self._last_episode_starts)
        callback.on_rollout_end()

        if self.save_ckpt:
            if len(self.ep_info_buffer) < self.n_envs:
                ret = [ep_info['r'] for ep_info in self.ep_info_buffer]
                ret = safe_mean(ret + [0.] * (self.n_envs - len(ret)))
            else:
                ret = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
            if not self.counter or self.highest_return < ret:
                self.highest_return = ret if not np.isnan(ret) else 0.0
                checkpoint = self._prepare_checkpoint()
                th.save(checkpoint, os.path.join(self.save_path, f'ckpt_best.pt'))
                print("Saved model checkpoint!!")

        return True

    def _prepare_checkpoint(self):
        ent_coef = self.ent_coef_schedule(self._current_progress_remaining, self.counter) \
            if self.ent_decay == 'exp' else self.ent_coef_schedule(self._current_progress_remaining)
        lr = self.lr_schedule(self._current_progress_remaining)
        clip_range = self.clip_range_schedule(self._current_progress_remaining)
        module_names = [d for d in dir(self.policy) if not d.startswith('_') and
                        isinstance(getattr(self.policy, d), torch.nn.Module) and d != 'model']
        checkpoint = {
            'network': {},
            'optimizer': self.policy.optimizer.state_dict(),
            'num_timesteps': self.num_timesteps,
            'ent_coef': ent_coef,
            'learning_rate': lr,
            'clip_range': clip_range
        }
        for m_name in module_names:
            checkpoint['network'][m_name] = getattr(self.policy, m_name).state_dict()
        checkpoint['seed'] = (np.random.get_state(), torch.get_rng_state(), random.getstate())
        return checkpoint

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "PPO",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, None, callback, 1, 100, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        latest_run = get_latest_run_id(self.tensorboard_log, 'PPO')
        self.save_path = os.path.join(os.path.join(self.tensorboard_log, f'PPO_{latest_run}'))
        callback.on_training_start(locals(), globals())
        self._dump_sources(self.save_path)
        self._dump_config(self.save_path)

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.n_steps)
            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    returns = [ep_info["r"] for ep_info in self.ep_info_buffer]
                    if len(returns) < self.n_envs:
                        returns = returns + [0] * (self.n_envs - len(returns))
                    self.logger.record("rollout/ep_rew_mean", safe_mean(returns))
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()
            self.counter += 1

        callback.on_training_end()

        return self
    
    def post_process(self, seg_num, instr_mask, instr):
        instr = instr.replace(",", "").split(' ')
        if self.find_then_before(int(instr_mask[seg_num][0]), instr):
            for t in range(seg_num):
                if instr_mask[t][1] == 0:
                    return False
            return True
        if self.find_after_after(int(instr_mask[seg_num][1]), instr):
            for t in range(seg_num + 1, 6):
                if instr_mask[t][1] == 0:
                    return False
            return True
        return True

class VideoEmbeddingModel(nn.Module):
    def __init__(self, embed_dim, num_heads, max_sequence_length, device):
        super(VideoEmbeddingModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length
        self.device = device
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads).to(device)
        self.positional_encodings = self._generate_positional_encodings(max_sequence_length, embed_dim)

    def forward(self, video_frames):
        # print(video_frames.shape)
        video_frames = video_frames + self.positional_encodings[:, :video_frames.size(1)].to(self.device)
        video_frames = video_frames.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        output, _ = self.multihead_attention(video_frames, video_frames, video_frames)
        output = output.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
        # average_embedding = torch.mean(output, dim=1)  # (batch_size, embed_dim)
        return output

    def _generate_positional_encodings(self, max_length, embed_dim):
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        positional_encodings = torch.zeros(1, max_length, embed_dim)
        positional_encodings[:, :, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, :, 1::2] = torch.cos(position * div_term)
        return positional_encodings
    
    
class DataBuffer:
    def __init__(self, num_envs, device, max_envs=5, max_steps=128):
        self.device = device
        self.buffers = [SingleBuffer(device, max_envs=max_envs, max_steps=max_steps) for i in range(num_envs)]
        
    def insert(self, vis, instr, reward, done):
        for env in range(len(self.buffers)):
            self.buffers[env].insert(vis[env], instr[env], reward[env], done[env])
            
    def get(self):
        max_len = 0
        for env in range(len(self.buffers)):
            length = self.buffers[env].get_max_len()
            max_len = max(max_len, length)
            
        films = []
        instrs = []
        rewards = []
        for env in range(len(self.buffers)):
            f, i, r = self.buffers[env].get(int(max_len))
            films.append(f)
            instrs.append(i)
            rewards.append(r)
        films = torch.cat(films, dim=0)
        instrs = np.concatenate(instrs, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        return films, instrs, rewards
    
    def is_full(self):
        for env in range(len(self.buffers)):
             if not self.buffers[env].is_full():
                 return False
        return True
    
    def get_first_frames(self, n):
        films, instrs = [], []
        
        max_n = 0
        for env in range(len(self.buffers)):
            new_n = self.buffers[env].get_current(n)
            if new_n > max_n:
                max_n = new_n
        
        for env in range(len(self.buffers)):
            f, i = self.buffers[env].get_first_frames(max_n)
            films.append(f)
            instrs.append(i)
        
        films = th.cat(films, dim=0)
        instrs = np.array(instrs)
        return films, instrs 
        
        
class SingleBuffer:
    def __init__(self, device, max_envs=5, max_steps=128):
        self.max_envs = max_envs
        self.counters = np.zeros(max_envs)
        self.current_film = 0
        self.films = th.zeros((max_envs, max_steps, 512), device=device)
        self.rewards = np.zeros((max_envs, max_steps))
        self.instrs = np.zeros((max_envs), dtype=object)
        
    def insert(self, vis, instr, reward, done):
        if self.current_film == self.max_envs:
            return
        if self.counters[self.current_film] == 0:
            self.instrs[self.current_film] = instr
        self.films[self.current_film] = vis
        self.rewards[self.current_film] = reward
        self.counters[self.current_film] += 1
        if done:
            self.current_film += 1
            
    def get(self, idx):  
        last_vid = self.current_film - 1
        return self.films[:last_vid, :idx, :], self.instrs[:last_vid], self.rewards[:last_vid, :idx]
    
    def get_max_len(self):
        return np.max(self.counters)
    
    def is_full(self):
        return self.current_film == self.max_envs
    
    def get_first_frames(self, max_len):
        films = self.films[self.index, :self.frame_index]
        films = th.nn.functional.pad(films, (0, 0, 0, max_len-self.frame_index))
        instr = self.instrs[self.index]
        # print(f'inside: {instr}')
        # print(f'index {self.index}')
        # print(self.instrs[0])
        # print(self.instrs[1])
        return th.unsqueeze(films, 0), instr
    
    def get_current(self, n):
        count = 0
        for env in range(self.max_envs):
            count += self.counters[env]
            if count >= n:
                count -= self.counters[env]
                self.index = env
                self.frame_index = int(n - count) + 1
                return self.frame_index
        return n
            