import torch.nn as nn
from torch.distributions import Categorical
from transformers import TransfoXLModel, TransfoXLConfig, TransfoXLTokenizer
import torch
import numpy as np
import clip
import os
from clip.simple_tokenizer import SimpleTokenizer
from transformers import AutoTokenizer
import math
import torch.nn.init as init


class DiscreteActor(nn.Module):
    def __init__(self, input_dim, hidden, out_dim, n_hidden=0):
        super(DiscreteActor, self).__init__()
        self.modlist = [nn.Linear(input_dim, hidden),
                        nn.LayerNorm(hidden, elementwise_affine=False),
                        nn.ReLU()]
        if n_hidden > 0:
            self.modlist.extend([nn.Linear(hidden, hidden),
                                 nn.LayerNorm(hidden, elementwise_affine=False),
                                 nn.ReLU()] * n_hidden)
        self.modlist.extend([nn.Linear(hidden, out_dim),
                            nn.Softmax(dim=-1)])
        self.actor = nn.Sequential(*self.modlist).apply(orthogonal_init)

    def forward(self, states, deterministic=False):
        probs = self.actor(states)
        dist = Categorical(probs)

        if deterministic:
            action = torch.argmax(probs).squeeze()
        else:
            action = dist.sample().squeeze()

        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate(self, states, actions):
        probs = self.actor(states)
        dist = Categorical(probs)
        log_prob = dist.log_prob(actions.squeeze())
        entropy = dist.entropy()
        return log_prob, entropy


class SmallImpalaCNN(nn.Module):
    def __init__(self, observation_shape, channel_scale=1, hidden_dim=256):
        super(SmallImpalaCNN, self).__init__()
        self.obs_size = observation_shape
        self.in_channels = 3
        kernel1 = 8 if self.obs_size[1] > 9 else 4
        kernel2 = 4 if self.obs_size[2] > 9 else 2
        stride1 = 4 if self.obs_size[1] > 9 else 2
        stride2 = 2 if self.obs_size[2] > 9 else 1
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=16*channel_scale, kernel_size=kernel1, stride=stride1),
                                    nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv2d(in_channels=16*channel_scale, out_channels=32*channel_scale, kernel_size=kernel2, stride=stride2),
                                    nn.ReLU())

        in_features = self._get_feature_size(self.obs_size)
        self.fc = nn.Linear(in_features=in_features, out_features=hidden_dim)

        self.hidden_dim = hidden_dim
        self.apply(xavier_uniform_init)

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            x = x.permute(0, 3, 1, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x

    def _get_feature_size(self, shape):
        if shape[0] != 3:
            dummy_input = torch.zeros((shape[-1], *shape[:-1])).unsqueeze(0)
            print(dummy_input.shape)
        else:
            dummy_input = torch.zeros((shape[0], *shape[1:])).unsqueeze(0)
        x = self.block2(self.block1(dummy_input))
        return np.prod(x.shape[1:])


class FrozenHopfield(nn.Module):
    def __init__(self, hidden_dim, input_dim, embeddings, beta):
        super(FrozenHopfield, self).__init__()
        self.rand_obs_proj = torch.nn.Parameter(torch.normal(mean=0.0, std=1 / np.sqrt(hidden_dim), size=(hidden_dim, input_dim)), requires_grad=False)
        self.word_embs = embeddings
        self.beta = beta

    def forward(self, observations):
        observations = self._preprocess_obs(observations)
        observations = observations @ self.rand_obs_proj.T
        similarities = observations @ self.word_embs.T / (
                    observations.norm(dim=-1).unsqueeze(1) @ self.word_embs.norm(dim=-1).unsqueeze(0) + 1e-8)
        softm = torch.softmax(self.beta * similarities, dim=-1)
        state = softm @ self.word_embs
        return state

    def _preprocess_obs(self, obs):
        obs = obs.mean(1)
        obs = torch.stack([o.view(-1) for o in obs])
        return obs


class HELM(nn.Module):
    def __init__(self, action_space, input_dim, optimizer, learning_rate, epsilon=1e-8, mem_len=511, beta=1,
                 device='cuda'):
        super(HELM, self).__init__()
        config = TransfoXLConfig()
        config.mem_len = mem_len
        self.mem_len = config.mem_len

        self.model = TransfoXLModel.from_pretrained('transfo-xl-wt103', config=config)
        n_tokens = self.model.word_emb.n_token
        word_embs = self.model.word_emb(torch.arange(n_tokens)).detach().to(device)
        hidden_dim = self.model.d_embed
        hopfield_input = np.prod(input_dim[1:])
        self.frozen_hopfield = FrozenHopfield(hidden_dim, hopfield_input, word_embs, beta=beta)

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.query_encoder = SmallImpalaCNN(input_dim, channel_scale=4, hidden_dim=hidden_dim)
        self.out_dim = hidden_dim*2
        self.actor = DiscreteActor(self.out_dim, 128, action_space.n).apply(orthogonal_init)
        self.critic = nn.Sequential(nn.Linear(self.out_dim, 512),
                                    nn.LayerNorm(512, elementwise_affine=False),
                                    nn.ReLU(),
                                    nn.Linear(512, 1)).apply(orthogonal_init)
        try:
            self.optimizer = getattr(torch.optim, optimizer)(self.yield_trainable_params(), lr=learning_rate,
                                                             eps=epsilon)
        except AttributeError:
            raise NotImplementedError(f"{optimizer} does not exist")
        self.memory = None

    def yield_trainable_params(self):
        for n, p in self.named_parameters():
            if 'model.' in n:
                continue
            else:
                yield p

    def forward(self, observations):
        bs, *_ = observations.shape
        obs_query = self.query_encoder(observations)
        vocab_encoding = self.frozen_hopfield.forward(observations)
        out = self.model(inputs_embeds=vocab_encoding.unsqueeze(1), output_hidden_states=True, mems=self.memory)
        self.memory = out.mems
        hidden = out.last_hidden_state[:, -1, :]
        hiddens = out.last_hidden_state[:, -1, :].cpu().numpy()

        hidden = torch.cat([hidden, obs_query], dim=-1)

        action, log_prob = self.actor(hidden)
        values = self.critic(hidden).squeeze()

        return action.cpu().numpy(), values.cpu().numpy(), log_prob.cpu().numpy().squeeze(), hiddens

    def evaluate_actions(self, hidden_states, actions, observations):
        queries = self.query_encoder(observations)
        hidden = torch.cat([hidden_states, queries], dim=-1)

        log_prob, entropy = self.actor.evaluate(hidden, actions)
        value = self.critic(hidden).squeeze()

        return value, log_prob, entropy


class HELMv2(nn.Module):
    def __init__(self, action_space, input_dim, optimizer, learning_rate, epsilon=1e-8, mem_len=511, device='cuda'):
        super(HELMv2, self).__init__()
        config = TransfoXLConfig()
        config.mem_len = mem_len
        self.mem_len = config.mem_len

        self.model = TransfoXLModel.from_pretrained('transfo-xl-wt103', config=config)
        n_tokens = self.model.word_emb.n_token
        word_embs = self.model.word_emb(torch.arange(n_tokens)).detach().to(device)
        self.we_std = word_embs.std(0)
        self.we_mean = word_embs.mean(0)
        self.vis_encoder = VisionBackbone("RN50")
        hidden_dim = self.model.d_embed

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.query_encoder = SmallImpalaCNN(input_dim, channel_scale=4, hidden_dim=hidden_dim)
        self.out_dim = hidden_dim*2
        self.actor = DiscreteActor(self.out_dim, 128, action_space.n).apply(orthogonal_init)
        self.critic = nn.Sequential(nn.Linear(self.out_dim, 512),
                                    nn.LayerNorm(512, elementwise_affine=False),
                                    nn.ReLU(),
                                    nn.Linear(512, 1)).apply(orthogonal_init)
        try:
            self.optimizer = getattr(torch.optim, optimizer)(self.yield_trainable_params(), lr=learning_rate,
                                                             eps=epsilon)
        except AttributeError:
            raise NotImplementedError(f"{optimizer} does not exist")
        self.memory = None

    def yield_trainable_params(self):
        for n, p in self.named_parameters():
            if 'model.' in n or 'vis_encoder' in n:
                continue
            else:
                yield p

    def forward(self, observations):
        bs, *_ = observations.shape
        obs_query = self.query_encoder(observations)
        observations = self.vis_encoder(observations)
        observations = (observations - observations.mean(0)) / (observations.std(0) + 1e-8)
        observations = observations * self.we_std + self.we_mean
        out = self.model(inputs_embeds=observations.unsqueeze(1), output_hidden_states=True, mems=self.memory)
        self.memory = out.mems
        hidden = out.last_hidden_state[:, -1, :]
        hiddens = out.last_hidden_state[:, -1, :].cpu().numpy()

        hidden = torch.cat([hidden, obs_query], dim=-1)

        action, log_prob = self.actor(hidden)
        values = self.critic(hidden).squeeze()

        return action.cpu().numpy(), values.cpu().numpy(), log_prob.cpu().numpy().squeeze(), hiddens

    def evaluate_actions(self, hidden_states, actions, observations):
        queries = self.query_encoder(observations)
        hidden = torch.cat([hidden_states, queries], dim=-1)

        log_prob, entropy = self.actor.evaluate(hidden, actions)
        value = self.critic(hidden).squeeze()

        return value, log_prob, entropy
    
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.fc1.bias)

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, input_dim)
        nn.init.eye_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual
        out = self.bn(out)
        return out

    
class TextEmbeddingModel(nn.Module):
    def __init__(self, embed_dim, num_heads, max_sequence_length, device='cpu'):
        super(TextEmbeddingModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(device)
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads).to(device)
        self.positional_encodings = self._generate_positional_encodings(max_sequence_length, embed_dim)

    def forward(self, text):
        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)['input_ids'].to(self.device)
        text_tokens = self.embedding(text_tokens)
        text_tokens = text_tokens + self.positional_encodings[:, :text_tokens.size(1)].to(self.device)
        text_tokens = text_tokens.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        output, _ = self.multihead_attention(text_tokens, text_tokens, text_tokens)
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

class SHELM(nn.Module):
    def __init__(self, action_space, input_dim, optimizer, learning_rate, env_id, topk=1, epsilon=1e-8, mem_len=511,
                 clip_encoder='ViT-B/16', device='cuda'):
        super(SHELM, self).__init__()
        config = TransfoXLConfig()
        config.mem_len = mem_len
        self.mem_len = config.mem_len
        self.device = device

        self.model = TransfoXLModel.from_pretrained('transfo-xl-wt103', config=config).to(device)
        # print(f'device: {device}')
        # self.resblock1 = ResidualBlock(input_dim=1024, hidden_dim=512).to(device)
        self.clip_tokenizer = SimpleTokenizer()
        self.tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        if 'psychlab' in env_id:
            self.clip_embs = np.load(os.path.join('data', f'{clip_encoder.replace("/", "")}_dmlab_prompt_embs.npz'))
        else:
            self.clip_embs = np.load(os.path.join('data', f'{clip_encoder.replace("/", "")}_embs.npz'))
        self.lexical_overlap = np.load(os.path.join('data', 'clip_transfo-xl-wt103_intersect.npz'))
        self.clip_embs = torch.FloatTensor(self.clip_embs[self.lexical_overlap]).cuda()
        n_tokens = self.model.word_emb.n_token
        self.word_embs = self.model.word_emb(torch.arange(n_tokens, device=device)).to(device)
        self.topk = topk

        self.vis_encoder = VisionBackbone(clip_encoder).to(device)
        self.resblock2 = ResidualBlock(input_dim=512, hidden_dim=256).to(device)
        hidden_dim = self.model.d_embed

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.query_encoder = SmallImpalaCNN(input_dim, channel_scale=4, hidden_dim=hidden_dim).to(device)
        self.out_dim = hidden_dim*3
        self.actor = DiscreteActor(self.out_dim, 128, action_space.n).apply(orthogonal_init)
        self.critic = nn.Sequential(nn.Linear(self.out_dim, 512),
                                    nn.LayerNorm(512, elementwise_affine=False),
                                    nn.ReLU(),
                                    nn.Linear(512, 1)).apply(orthogonal_init)
        self.instr_att = TextEmbeddingModel(hidden_dim, 1, 32, device)
        self.ll = torch.nn.Linear(512, 1024).to(device)
        self.action_embedding = nn.Embedding(num_embeddings=7, embedding_dim=512).to(device)
        
        try:
            self.optimizer = getattr(torch.optim, optimizer)(self.yield_trainable_params(), lr=learning_rate,
                                                             eps=epsilon)
        except AttributeError:
            raise NotImplementedError(f"{optimizer} does not exist")
        self.memory = None

    def yield_trainable_params(self):
        for n, p in self.named_parameters():
            if 'model.' in n or 'vis_encoder' in n:
                continue
            else:
                yield p

    def _calc_cos_sim(self, src, target):
        normed_src = src / src.norm(dim=-1, keepdim=True)
        normed_tar = target / target.norm(dim=-1, keepdim=True)
        return normed_src @ normed_tar.T

    def _get_top_k_toks(self, src, tar, k=1):
        cos_sims = self._calc_cos_sim(src, tar)
        ranked = np.argsort(cos_sims.detach().cpu().numpy(), axis=-1)[:, ::-1][:, :k]
        ranked = self.lexical_overlap[ranked]
        decoded = []
        embs = []
        for toks in ranked:
            dec = [self.clip_tokenizer.decode([t]) for t in toks]
            decoded.append(dec)
            enc = self.tokenizer.encode(dec)
            embs.append(self.word_embs[enc])
        embs = torch.stack(embs)
        return embs, decoded

    def forward(self, observations, instr):
        if observations.shape[1] != 3:
            bs, h, w, c = observations.shape
            observations = observations.reshape(bs, c, h, w)
        else:
            bs, *_ = observations.shape
        obs_query = self.query_encoder(observations)
        observations = self.vis_encoder(observations)
        observations, _ = self._get_top_k_toks(observations, self.clip_embs, self.topk)
        if len(observations.shape) == 2:
            observations = observations.unsqueeze(1)
        out = self.model(inputs_embeds=observations, output_hidden_states=True, mems=self.memory)
        self.memory = out.mems
        hidden = out.last_hidden_state[:, -1, :]
        hiddens = out.last_hidden_state[:, -1, :].cpu().numpy()

        instr = instr.tolist()
        instr_embeds = self.instr_att(instr)
        instr_embed = torch.mean(instr_embeds, dim=1)
        
        hidden = torch.cat([hidden, instr_embed, obs_query], dim=-1)

        action, log_prob = self.actor(hidden)
        values = self.critic(hidden).squeeze()

        return action.cpu().numpy(), values.cpu().numpy(), log_prob.cpu().numpy().squeeze(), hiddens
    
    def clip_forward(self, observations, actions):
        if observations.shape[1] != 3:
            bs, h, w, c = observations.shape
            observations = observations.reshape(bs, c, h, w)
        else:
            bs, *_ = observations.shape
        observations = self.vis_encoder(observations)
        # observations = self.query_encoder(observations)
        actions = torch.tensor(actions).to(self.device)
        actions = self.action_embedding(actions)
        observations = torch.add(observations, actions)
        return  observations
    
    def get_instr_embeddings(self, instrs):
        embeddings = []
        for i in range(len(instrs)):
            instr = instrs[i].tolist()
            instr_embed = self.instr_att(instr)
            embeddings.append(instr_embed)
        embeddings = torch.stack(embeddings)
        return  torch.squeeze(embeddings)
    
    def up_project(self, frame_embeddings):
        shape = frame_embeddings.shape
        frame_embeddings = frame_embeddings.reshape(shape[0] * shape[1], shape[2])
        embeddings = self.ll(frame_embeddings)
        embeddings = embeddings.reshape(shape[0], shape[1], -1)
        return  embeddings

    def evaluate_actions(self, hidden_states, actions, observations, instr):
        instr = instr.tolist()
        instr_embeds = self.instr_att(instr)
        instr_embed = torch.mean(instr_embeds, dim=1)
        if observations.shape[1] != 3:
            bs, h, w, c = observations.shape
            observations = observations.reshape(bs, c, h, w)
        else:
            bs, *_ = observations.shape
        queries = self.query_encoder(observations)
        hidden = torch.cat([hidden_states, instr_embed, queries], dim=-1)

        log_prob, entropy = self.actor.evaluate(hidden, actions)
        value = self.critic(hidden).squeeze()

        return value, log_prob, entropy


class VisionBackbone(nn.Module):
    def __init__(self, encoder):
        super(VisionBackbone, self).__init__()
        print(f"Allocating CLIP...")
        self.model, preprocess = clip.load(encoder)
        self.transforms = preprocess
        preprocess.transforms = [preprocess.transforms[0], preprocess.transforms[1], preprocess.transforms[-1]]
        self.transforms = preprocess
        self.n_channels = 3

        self.model.eval()
        self._deactivate_grad()

    def forward(self, observations):
        if observations.shape[1] != self.n_channels:
            observations = observations.permute(0, 3, 1, 2)
        observations = self._preprocess(observations)
        out = self.model.encode_image(observations).float()
        return out

    def _deactivate_grad(self):
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _preprocess(self, observation):
        return self.transforms(observation)


class MarkovianImpalaCNN(nn.Module):
    def __init__(self, obs_dim, action_dim, optimizer, learning_rate):
        super(MarkovianImpalaCNN, self).__init__()
        self.encoder = SmallImpalaCNN(obs_dim, channel_scale=4, hidden_dim=1024)
        hidden_dim = self.encoder.hidden_dim

        self.actor = DiscreteActor(hidden_dim, hidden=128, out_dim=action_dim).apply(orthogonal_init)

        critic_modules = []
        critic_modules.extend([nn.Linear(hidden_dim, 512),
                               nn.LayerNorm(512, elementwise_affine=False),
                               nn.ReLU()])
        critic_modules.append(nn.Linear(512, 1))
        self.critic = nn.Sequential(*critic_modules).apply(orthogonal_init)

        try:
            self.optimizer = getattr(torch.optim, optimizer)(self.parameters(), lr=learning_rate)
        except AttributeError:
            raise NotImplementedError(f"{optimizer} does not exist")

    def forward(self, states):
        encoded = self.encoder(states)
        action, log_prob = self.actor(encoded)
        value = self.critic(encoded)
        return action.cpu().detach().numpy(), value.cpu().detach().squeeze().numpy(), log_prob.cpu().detach().numpy()

    def evaluate_actions(self, states, actions):
        encoded = self.encoder(states)
        log_probs, entropy = self.actor.evaluate(encoded, actions)
        values = self.critic(encoded).squeeze()
        return values, log_probs, entropy


class LSTMImpalaAgent(nn.Module):
    def __init__(self, action_dim, input_dim, optimizer, learning_rate, channel_scale=1, hidden_dim=256):
        super(LSTMImpalaAgent, self).__init__()
        self.encoder = SmallImpalaCNN(input_dim, channel_scale=channel_scale, hidden_dim=hidden_dim)
        self.hidden_dim = self.encoder.hidden_dim
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.actor = DiscreteActor(self.hidden_dim, 128, action_dim).apply(orthogonal_init)
        self.critic = nn.Sequential(nn.Linear(self.hidden_dim, 512),
                                    nn.LayerNorm(512, elementwise_affine=False),
                                    nn.ReLU(),
                                    nn.Linear(512, 1)).apply(orthogonal_init)
        self.hidden = None
        self.cell = None
        try:
            self.optimizer = getattr(torch.optim, optimizer)(self.parameters(), lr=learning_rate)
        except AttributeError:
            raise NotImplementedError(f"{optimizer} does not exist")

    def reset_states(self):
        self._init_hidden(1)

    def _init_hidden(self, batch_size):
        device = next(self.parameters()).device
        self.hidden = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        self.cell = torch.zeros(1, batch_size, self.hidden_dim).to(device)

    def forward(self, state):
        bs, *_ = state.shape
        encoded = self.encoder(state)
        if self.hidden is None and self.cell is None:
            self._init_hidden(bs)
        last_hidden = np.array([self.hidden.cpu().numpy().squeeze(), self.cell.cpu().numpy().squeeze()])
        hidden, (self.hidden, self.cell) = self.lstm(encoded.unsqueeze(1), (self.hidden, self.cell))
        hidden = hidden[:, -1, :]
        action, log_prob = self.actor(hidden)
        values = self.critic(hidden).squeeze()
        return action.cpu().numpy(), values.cpu().numpy(), log_prob.cpu().numpy().squeeze(), last_hidden

    def evaluate_actions(self, states, actions, internals, detach_value_grad=False):
        bs, seqlen, *_ = states.shape
        states = states.reshape(bs*seqlen, *states.shape[2:])
        encoded = self.encoder(states)
        encoded = encoded.view(bs, seqlen, -1)
        internals = (internals[:, 0, 0, :].unsqueeze(0).contiguous(), internals[:, 0, 1, :].unsqueeze(0).contiguous())
        hidden, _ = self.lstm(encoded, internals)
        log_prob, entropy = self.actor.evaluate(hidden, actions)
        if detach_value_grad:
            hidden = hidden.detach()
        value = self.critic(hidden)
        return value, log_prob, entropy


def orthogonal_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        torch.nn.init.orthogonal_(m.weight.data, gain=gain)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)


def xavier_uniform_init(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0.)
    return module
