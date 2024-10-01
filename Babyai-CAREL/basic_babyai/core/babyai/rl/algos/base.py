from abc import ABC, abstractmethod
import torch
import numpy
from operator import add
import torch.nn as nn
import torch.nn as nn
from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList, ParallelEnv
from babyai.rl.utils.supervised_losses import ExtraInfoCollector
import matplotlib.pyplot as plt
import sys
import re

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm,  recurrence, preprocess_obss, 
                 reshape_reward, aux_info, use_compositional_split=False, 
                 compositional_test_splits=None, device=None, apply_instruction_tracking=False, threshold=2):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        aux_info : list
            a list of strings corresponding to the name of the extra information
            retrieved from the environment for supervised auxiliary losses

        """
        # Store parameters

        self.env = ParallelEnv(envs, 
            use_compositional_split=use_compositional_split, 
            compositional_test_splits=compositional_test_splits)
        self.acmodel = acmodel
        self.acmodel.train()
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.aux_info = aux_info
        self.use_compositional_split = use_compositional_split
        self.compositional_test_splits = compositional_test_splits
        self.apply_instruction_tracking = apply_instruction_tracking
        self.threshold = threshold

        # Store helpers values

        self.device = device
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs


        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])

        self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
        self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)

        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)
        if self.aux_info:
            self.aux_info_collector = ExtraInfoCollector(self.aux_info, shape, self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs
        if apply_instruction_tracking:
            # self.ln = nn.Linear(1, 64).to(self.device)
            self.action_embedding = nn.Embedding(num_embeddings=self.env.action_space.n, embedding_dim=64).to(self.device)
            self.counter = 0

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        
        self.acmodel.reset_hiddens()
        subtask_segments_scores = numpy.zeros((self.num_procs, 5, self.num_frames_per_proc))
        video_begining = numpy.zeros((self.num_procs), dtype=numpy.int64)
        instr_mask = numpy.zeros((self.num_procs, 5, 2))
        for i in range(self.num_frames_per_proc):
            # count over frames processed (used for scheduling)
            self.counter += self.num_procs
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                model_results = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                dist = model_results['dist']
                value = model_results['value']
                memory = model_results['memory']
                extra_predictions = model_results['extra_predictions']

            action = dist.sample()
            # print(f'number of actions: {self.env.action_space.n}')
            obs, reward, done, env_info = self.env.step(action.cpu().numpy())

            # split task to subtasks
            if self.apply_instruction_tracking:
                observation_instr = [0] * self.num_procs
                for agent_id in range(self.num_procs):
                    total_instr = obs[agent_id]['mission']
                    total_instr = total_instr.replace(",", "")
                    instr_tokens = total_instr.split(' ')
                    instr_tokens = [idx for idx, x in enumerate(instr_tokens) if x in ['and', 'then', 'after', 'before', 'or']]
                    observation_instr[agent_id] = instr_tokens
            
            if self.aux_info:
                env_info = self.aux_info_collector.process(env_info)
                # env_info = self.process_aux_info(env_info)

            # Update experiences values
            
            # reset masks if done
            if self.apply_instruction_tracking:
                for proc in range(self.num_procs):
                    if done[proc]:
                        instr_mask[proc] = numpy.zeros((5, 2))

            self.obss[i] = self.obs
            self.obs = obs

            self.memories[i] = self.memory
            self.memory = memory

            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value

            self.rewards[i] = torch.tensor(reward, device=self.device)

            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, self.rewards[i], done)
                ], device=self.device)
                        
            # instr masking
            if self.apply_instruction_tracking:
                # each episode can consist of some successful trajectories
                # we save the start of each new trajectory
                # truncate and pad observation, actions and mask accoring to beginging of trajectories
                truncated_obs = [self.obss[k][j]
                        for j in range(self.num_procs)
                        for k in range(numpy.min(video_begining), i+1)]
                truncated_obs = self.preprocess_obss(truncated_obs, device=self.device)
                truncated_mask = self.masks[numpy.min(video_begining):i+1].transpose(0, 1).reshape(-1).unsqueeze(1)
                aux_actions = self.actions[numpy.min(video_begining):i+1].transpose(0, 1)
                
                # prepare action embedding to add to embedding matrix
                action_shape = aux_actions.shape
                aux_actions = aux_actions.reshape(-1, 1)
                aux_actions = self.action_embedding(aux_actions.to(torch.long)).reshape(*action_shape, 64)      

                num_frames = truncated_mask.shape[0] // self.num_procs
                truncated_mask = truncated_mask.reshape(self.num_procs, -1, 1)
                
                # pad frames
                truncated_obs, truncated_mask, aux_actions = self.pad_frames(truncated_obs, truncated_mask, aux_actions, video_begining, num_frames)

                truncated_mem = self.memories[numpy.min(video_begining):i+1].transpose(0, 1).reshape(-1, *self.memories.shape[2:])
                model_results = self.acmodel(truncated_obs, truncated_mem * truncated_mask, mask=truncated_mask)
                # extract embedding matrices
                with torch.no_grad():
                    text_matrix = model_results["token_embedding"]
                    text_matrix = text_matrix.reshape(self.num_procs, num_frames, *text_matrix.shape[1:])[:, num_frames-1]
                    video_matrix = model_results["frame_embedding"]
                    video_matrix = video_matrix.reshape(self.num_procs, num_frames, *video_matrix.shape[1:])
                    video_matrix = torch.add(video_matrix, aux_actions)
                    video_global_embeddings = self.video_attn_model(video_matrix)

                    for proc in range(self.num_procs):
                        # instr token to whole video matrix
                        token_video_score = torch.matmul(text_matrix[proc], video_global_embeddings[proc]).T
                        split_list = observation_instr[proc]
                        if not split_list:
                            continue
                        split_list.insert(0, -1)
                        for idx in range(len(split_list)): 
                            # calculate start and end of each subtask
                            start_index = split_list[idx] + 1   
                            if idx == len(split_list)-1:
                                end_index = len(self.obs[proc]['mission'].replace(",", "").split(' '))
                            else:
                                end_index = split_list[idx+1]
                            
                            segment_score1 = numpy.mean(token_video_score.cpu().detach().numpy()[start_index:end_index])
                            # subtask_embedding = torch.tensor(self.preprocess_obss.instr_preproc([{'mission': 
                            #     ' '.join(self.obs[proc]['mission'].replace(",", "").split(' ')[start_index:end_index])}]), device=self.device)
                            # subtask_global_embedding = self.acmodel._get_instr_embedding(subtask_embedding)[0][0]
                            # segment_score2 = torch.matmul(video_global_embeddings[proc].T, subtask_global_embedding) # global video and global instruction score candidate
                            # semantic_score3 = numpy.max(torch.matmul(video_matrix[proc], subtask_global_embedding).cpu().detach().numpy()) # frame and subtask score candidate
                            segment_score = segment_score1
                            # append score for that subtask
                            subtask_segments_scores[proc, idx, i] = segment_score
                            
                            # check if masking needed
                            if segment_score > self.threshold * numpy.mean(subtask_segments_scores[proc, idx, :i-1]):
                                # mask with a probabilty
                                prob = 0.8 * numpy.tanh(self.counter/1e7) + 0.01
                                if prob >= numpy.random.uniform(0, 1):
                                    instr_mask[proc][idx][0], instr_mask[proc][idx][1] = start_index, end_index


                #### Applying Mask ####
                # only apply masking after forth frame
                if i >= 4:
                    for proc in range(self.num_procs):
                        for seg in range(5):
                            if instr_mask[proc][seg][1] != 0:
                                if self.post_process(seg, instr_mask[proc], self.obs[proc]['mission']):
                                    instr = self.obs[proc]['mission'].replace(",", "").split(' ')
                                    for l in range(int(instr_mask[proc][seg][0]), int(instr_mask[proc][seg][1])):
                                        instr[l] = '<mask>'
                                    instr = ' '.join(instr)
                                    self.obs[proc]['mission'] = instr
                                else:
                                    instr_mask[proc][seg][0], instr_mask[proc][seg][1] = 0, 0

                # reset begining index if done
                for proc in range(self.num_procs):
                    if done[proc]:
                        video_begining[proc] = i+1   

                  
            self.log_probs[i] = dist.log_prob(action)
            

            if self.aux_info:
                self.aux_info_collector.fill_dictionaries(i, env_info, extra_predictions)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            next_value = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))['value']

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        exps = DictList()
        obss = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        # In commments below T is self.num_frames_per_proc, P is self.num_procs,
        # D is the dimensionality

        # T x P x D -> P x T x D -> (P * T) x D
        exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        if self.aux_info:
            exps = self.aux_info_collector.end_collection(exps)

        # Preprocess experiences
        exps.obs = self.preprocess_obss(obss, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "episodes_done": self.log_done_counter,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, log, obss

    @abstractmethod
    def update_parameters(self):
        pass
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
    
    def find_then_before(self, index, instr):
        return 'then' in instr[:index]
    
    def find_after_after(self, index, instr):
        return 'after' in instr[index: ]

    def convert_instr(self, instr):
        pattern = r"put the (\w+) (\w+) next to the (\w+) (\w+)"
    
        matches = re.findall(pattern, instr)
        color1, type1, color2, type2 = matches[0]
        output = f"go to {color1} {type1}, then pickup {color1} {type1}, then go to {color2} {type2}"
    
        return output
    
    def pad_frames(self, truncated_obs, truncated_mask, aux_actions, video_begining, num_frames):
        for proc in range(self.num_procs):
                    diff = video_begining[proc] - numpy.min(video_begining)
                    aux_actions[proc][:diff] = torch.zeros((diff, 64))
                    truncated_mask[proc][:diff] = torch.zeros((diff, 1))
                    for d in range(diff):
                        truncated_obs.__setitem__(num_frames * proc + d, {'image': torch.zeros_like(truncated_obs[num_frames * proc + d].image, device=self.device), 
                                                                    'instr': torch.zeros_like(truncated_obs[num_frames * proc + d].instr, device=self.device)}) 
        truncated_mask = truncated_mask.reshape(-1, 1)
        return truncated_obs, truncated_mask, aux_actions 