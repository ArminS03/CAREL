import sys
import os
from pathlib import Path


import torch
import torch.nn as nn
import torch.nn.functional as F
import babyai.rl
import time
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from babyai.rl.utils.supervised_losses import required_heads

from babyai.model import ExpertControllerFiLM

from collections import defaultdict


def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1 and classname.find("Group") == -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, babyai.rl.RecurrentACModel):
    def __init__(
        self,
        obs_space,
        action_space,
        memory_dim=64,
        instr_dim=64,
        use_instr=False,
        lang_model="gru",
        use_memory=False,
        arch="cnn1",
        aux_info=None,
        film_d=128,
        device="cpu",
    ):
        super().__init__()

        # Decide which components are enabled
        self.use_instr = use_instr
        self.use_memory = use_memory
        self.arch = arch
        self.lang_model = lang_model
        self.aux_info = aux_info
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim
        self.device = device
        self.obs_space = obs_space

        self.x_clip = nn.Sequential(
            nn.Linear(in_features=1024, out_features=self.instr_dim), nn.GELU()
        )

        if arch == "cnn1":
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=memory_dim, kernel_size=(2, 2)),
                nn.ReLU(),
            )
        elif arch.startswith("expert_filmcnn"):
            if not self.use_instr:
                raise ValueError(
                    "FiLM architecture can be used when instructions are enabled"
                )

            self.image_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=3, out_channels=film_d, kernel_size=(2, 2), padding=1
                ),
                nn.BatchNorm2d(film_d),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(
                    in_channels=film_d,
                    out_channels=film_d,
                    kernel_size=(3, 3),
                    padding=1,
                ),
                nn.BatchNorm2d(film_d),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            )
            self.film_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        else:
            raise ValueError("Incorrect architecture name: {}".format(arch))

        # Define instruction embedding
        self.final_instr_dim = instr_dim
        if self.use_instr or self.concat_instr_to_mem or self.input_mem_concat_instr:
            if self.lang_model in ["gru", "bigru", "attgru"]:
                self.word_embedding = nn.Embedding(
                    self.obs_space["instr"], self.instr_dim, padding_idx=0
                )
                if self.lang_model in ["gru", "bigru", "attgru"]:
                    gru_dim = self.instr_dim
                    if self.lang_model in ["bigru", "attgru"]:
                        gru_dim //= 2
                    self.instr_rnn = nn.GRU(
                        self.instr_dim,
                        gru_dim,
                        batch_first=True,
                        bidirectional=(self.lang_model in ["bigru", "attgru"]),
                    )
                    self.final_instr_dim = self.instr_dim
                else:
                    kernel_dim = 64
                    kernel_sizes = [3, 4]
                    self.instr_convs = nn.ModuleList(
                        [
                            nn.Conv2d(1, kernel_dim, (K, self.instr_dim))
                            for K in kernel_sizes
                        ]
                    )
                    self.final_instr_dim = kernel_dim * len(kernel_sizes)

            if self.lang_model == "attgru":
                self.memory2key = nn.Linear(self.memory_size, self.final_instr_dim)

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.memory_dim, self.memory_dim)

        # # Resize image embedding
        self.embedding_size = self.semi_memory_size

        if arch.startswith("expert_filmcnn"):
            if arch == "expert_filmcnn":
                num_module = 2
            else:
                num_module = int(arch[(arch.rfind("_") + 1) :])
                print("num_controllers: ", num_module)
            self.controllers = []
            for ni in range(num_module):
                if ni < num_module - 1:
                    mod = ExpertControllerFiLM(
                        in_features=self.final_instr_dim,
                        out_features=film_d,
                        in_channels=film_d,
                        imm_channels=film_d,
                    )
                else:
                    mod = ExpertControllerFiLM(
                        in_features=self.final_instr_dim,
                        out_features=self.memory_dim,
                        in_channels=film_d,
                        imm_channels=film_d,
                    )
                self.controllers.append(mod)
                self.add_module("FiLM_Controler_" + str(ni), mod)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size+64 if not "filmcnn" in self.arch else self.embedding_size, 64), nn.Tanh(), nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size+64 if not "filmcnn" in self.arch else self.embedding_size, 64), nn.Tanh(), nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Define head for extra info
        if self.aux_info:
            self.extra_heads = None
            self.add_heads()

    def reset_hiddens(self):
        pass

    def add_heads(self):
        """
        When using auxiliary tasks, the environment yields at each step some binary, continous, or multiclass
        information. The agent needs to predict those information. This function add extra heads to the model
        that output the predictions. There is a head per extra information (the head type depends on the extra
        information type).
        """
        self.extra_heads = nn.ModuleDict()
        for info in self.aux_info:
            if required_heads[info] == "binary":
                self.extra_heads[info] = nn.Linear(self.embedding_size, 1)
            elif required_heads[info].startswith("multiclass"):
                n_classes = int(required_heads[info].split("multiclass")[-1])
                self.extra_heads[info] = nn.Linear(self.embedding_size, n_classes)
            elif required_heads[info].startswith("continuous"):
                if required_heads[info].endswith("01"):
                    self.extra_heads[info] = nn.Sequential(
                        nn.Linear(self.embedding_size, 1), nn.Sigmoid()
                    )
                else:
                    raise ValueError("Only continous01 is implemented")
            else:
                raise ValueError("Type not supported")
            # initializing these parameters independently is done in order to have consistency of results when using
            # supervised-loss-coef = 0 and when not using any extra binary information
            self.extra_heads[info].apply(initialize_parameters)

    def add_extra_heads_if_necessary(self, aux_info):
        """
        This function allows using a pre-trained model without aux_info and add aux_info to it and still make
        it possible to finetune.
        """
        try:
            if not hasattr(self, "aux_info") or not set(self.aux_info) == set(aux_info):
                self.aux_info = aux_info
                self.add_heads()
        except Exception:
            raise ValueError("Could not add extra heads")

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def add_obs_count(self, obs, mask):
        counts = []
        for i in range(obs.shape[0]):
            o = obs[i]
            m = mask[i]
            k = "".join(
                list(o.detach().cpu().numpy().flatten().astype(int).astype(str))
            )
            if not m:  # mask is 1 - done.
                self.intrinsic_rew_counts[k] = 0
            self.intrinsic_rew_counts[k] += 1
            counts.append(self.intrinsic_rew_counts[k])
        return torch.tensor(counts).to(self.device)

    def forward(self, obs, memory, instr_embedding=None, mask=None):
        vq_loss = None
        residual_instr = None
        if self.use_instr and instr_embedding is None:
            instr_embedding, instr_tokens = self._get_instr_embedding(obs.instr)
            if self.lang_model == "gru":
                residual_instr = instr_embedding
        if self.use_instr and self.lang_model == "attgru":
            # outputs: B x L x D
            # memory: B x M
            mask = (obs.instr != 0).float()
            instr_embedding = instr_embedding[:, : mask.shape[1]]
            keys = self.memory2key(memory)
            pre_softmax = (keys[:, None, :] * instr_embedding).sum(2) + 1000 * mask
            attention = F.softmax(pre_softmax, dim=1)
            instr_embedding = (instr_embedding * attention[:, :, None]).sum(1)

        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        intrinsic_rew = torch.zeros((obs.image.shape[0],)).to(self.device)
        if self.arch.startswith("expert_filmcnn"):
            x = self.image_conv(x)
            middle_x = x
            for controler in self.controllers:
                x = controler(x, instr_embedding)
            x = F.relu(self.film_pool(x))
        else:
            x = self.image_conv(x)
            middle_x = x

        x = x.reshape(x.shape[0], -1)
        frame_embedding = x
        frame_embedding = self.x_clip(frame_embedding.to(self.device))

        if self.use_memory:
            bs, mh = memory.shape
            hidden = (
                memory[:, : self.semi_memory_size],
                memory[:, self.semi_memory_size :],
            )
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if not "filmcnn" in self.arch:
            embedding = torch.nn.functional.normalize(embedding, dim=1)
            residual_instr = torch.nn.functional.normalize(residual_instr, dim=1)
            embedding = torch.cat((embedding, residual_instr), dim=1)

        if hasattr(self, "aux_info") and self.aux_info:
            extra_predictions = {
                info: self.extra_heads[info](embedding) for info in self.extra_heads
            }
        else:
            extra_predictions = dict()

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=-1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return {
            "dist": dist,
            "value": value,
            "memory": memory,
            "extra_predictions": extra_predictions,
            "middle_reps": dict(middle_x=middle_x),
            "m_ts": None,
            "frame_embedding": frame_embedding,
            "token_embedding": instr_tokens,
            "instr_embedding": instr_embedding
        }

    def _get_instr_embedding(self, instr):
        if self.lang_model == "gru":
            embedded = self.word_embedding(instr).to(self.device)
            _, hidden = self.instr_rnn(embedded)
            return hidden[-1], embedded

        elif self.lang_model in ["bigru", "attgru"]:
            raise "Not implemented properly!"

        else:
            ValueError("Undefined instruction architecture: {}".format(self.use_instr))
