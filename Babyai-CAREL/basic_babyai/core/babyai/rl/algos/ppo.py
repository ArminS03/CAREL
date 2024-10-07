from os import device_encoding
import math
import math
import sys
import numpy
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F



from babyai.rl.algos.base import BaseAlgo


def NS_single_step(pred, target):
    metric = nn.MSELoss()
    bs = pred.shape[0]
    return metric(pred, target.reshape(bs, -1))


class PPOAlgo(BaseAlgo):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(
        self,
        envs,
        acmodel,
        num_frames_per_proc=None,
        discount=0.99,
        lr=7e-4,
        beta1=0.9,
        beta2=0.999,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        recurrence=4,
        adam_eps=1e-5,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        preprocess_obss=None,
        reshape_reward=None,
        aux_info=None,
        use_compositional_split=False,
        compositional_test_splits=None,
        device=None,
        # CAREL
        apply_instruction_tracking=False,
        threshold=2,
        apply_aux=False,
        x_clip_coef=1,
        x_clip_temp=1,
        num_attn_heads=4,
        att_dim=64,
    ):
        self.x_clip_coef = x_clip_coef
        self.x_clip_temp = x_clip_temp
        self.apply_aux = apply_aux
        
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(
            envs,
            acmodel,
            num_frames_per_proc,
            discount,
            lr,
            gae_lambda,
            entropy_coef,
            value_loss_coef,
            max_grad_norm,
            recurrence,
            preprocess_obss,
            reshape_reward,
            aux_info,
            use_compositional_split=use_compositional_split,
            compositional_test_splits=compositional_test_splits,
            device=device,
            apply_instruction_tracking=apply_instruction_tracking,
            threshold=threshold,
        )

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        self.batch_num = 0
        self.att_dim = att_dim
        self.num_attn_heads = num_attn_heads
        self.video_attn_model = VideoEmbeddingModel(att_dim, self.num_attn_heads, num_frames_per_proc, device)
        self.optimizer = torch.optim.Adam(
            list(self.acmodel.parameters()) + list(self.video_attn_model.parameters()), lr, (beta1, beta2), eps=adam_eps
        )


    def update_parameters(self):
        # Collect experiences

        exps, logs, obss = self.collect_experiences()
        """
        exps is a DictList with the following keys ['obs', 'memory', 'mask', 'action', 'value', 'reward',
         'advantage', 'returnn', 'log_prob'] and ['collected_info', 'extra_predictions'] if we use aux_info
        exps.obs is a DictList with the following keys ['image', 'instr']
        exps.obs.image is a (n_procs * n_frames_per_proc) x image_size 4D tensor
        exps.obs.instr is a (n_procs * n_frames_per_proc) x (max number of words in an instruction) 2D tensor
        exps.memory is a (n_procs * n_frames_per_proc) x (memory_size = 2*image_embedding_size) 2D tensor
        exps.mask is (n_procs * n_frames_per_proc) x 1 2D tensor
        if we use aux_info: exps.collected_info and exps.extra_predictions are DictLists with keys
        being the added information. They are either (n_procs * n_frames_per_proc) 1D tensors or
        (n_procs * n_frames_per_proc) x k 2D tensors where k is the number of classes for multiclass classification
        """
        
        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            log_losses = []
            x_clip_losses = []
            
            self.acmodel.reset_hiddens()
            
            #################################################################################################
            # Preprocessing
            if self.apply_aux:
                ## detect successful episodes
                masks = exps.mask.detach().clone().cpu().numpy()
                completed_videos = numpy.where(masks == 0)[0]
                video_info = numpy.array([[i // self.num_frames_per_proc, i % self.num_frames_per_proc] for i in completed_videos])
                

                num_samples = 0
                # index of successful episode
                video_idx = numpy.array([], dtype=int)
                # index of termination of an episode
                video_len = numpy.array([], dtype=int)
                max_len = 0

                if video_info.size != 0:
                    for idx in video_info:
                        if idx[0] not in video_idx:
                            video_idx = numpy.append(video_idx, idx[0])
                            video_len = numpy.append(video_len, idx[1])
                    num_samples = video_idx.shape[0]
                    max_len = numpy.max(video_len) + 1

                # X-CLIP loss
                if num_samples > 1:
                    similarity_mx = torch.zeros((num_samples, num_samples), device=self.device)
                    video_matrix = torch.zeros(
                        (num_samples, max_len, self.acmodel.instr_dim), device=self.device
                    )
                    text_matrix = torch.zeros(
                        (num_samples, exps.obs.instr.shape[1], self.acmodel.instr_dim), device=self.device
                    )
                    text_global_embeddings = torch.zeros(
                        (num_samples, self.acmodel.instr_dim), device=self.device
                    )
                    # build feature matrices
                    env_start_index = video_idx * self.num_frames_per_proc 
                    for i in range(len(env_start_index)):
                        idx = env_start_index[i]
                        env_idx = range(idx, idx + max_len)
                        sb = exps[env_idx]
                        model_results = self.acmodel(sb.obs, sb.memory * sb.mask, mask=sb.mask)
                        # get token and sentence embeddings
                        text_matrix[i] = model_results["token_embedding"][0]
                        text_global_embeddings[i] = model_results["instr_embedding"][0]
                        # get frame embeddigns for a single frame
                        video_matrix[i] = model_results["frame_embedding"]

                    for i in range(len(video_len)):
                        video_matrix[i, video_len[i]+1:, :] = 0
                    video_global_embeddings = self.video_attn_model(video_matrix)

                    #calculate similarity matrix
                    for i in range(num_samples):
                        for j in range(num_samples):
                            frame_token_similarity = self.Attention_Over_Similarity_Matrix(
                                torch.matmul(video_matrix[i], text_matrix[j].T), self.x_clip_temp
                            )
                            text_frame_similarity = self.Attention_Over_Similarity_Vector(
                                torch.matmul(video_matrix[i], text_global_embeddings[j]), self.x_clip_temp
                            )
                            video_token_similarity = self.Attention_Over_Similarity_Vector(
                                torch.matmul(text_matrix[j], video_global_embeddings[i]).T, self.x_clip_temp
                            )
                            video_text_similarity = torch.matmul(
                                video_global_embeddings[i].T, text_global_embeddings[j]
                            )
                            similarity_mx[i][j] = (
                                frame_token_similarity
                                + text_frame_similarity
                                + video_token_similarity
                                + video_text_similarity
                            ) / 4
                    # calculate loss function and optimize 
                    x_clip_loss = self.calculate_contrastive_loss(similarity_mx) * self.x_clip_coef
                    x_clip_losses.append(x_clip_loss.detach().item())
                    self.optimizer.zero_grad()
                    x_clip_loss.backward()
                    self.optimizer.step()          
                #################################################################################################

            """
            For each epoch, we create int(total_frames / batch_size + 1) batches, each of size batch_size (except
            maybe the last one). Each batch is divided into sub-batches of size recurrence (frames are contiguous in
            a sub-batch), but the position of each sub-batch in a batch and the position of each batch in the whole
            list of frames is random thanks to self._get_batches_starting_indexes().
            """
            for inds in self._get_batches_starting_indexes():
                # inds is a numpy array of indices that correspond to the beginning of a sub-batch
                # there are as many inds as there are batches
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                memory = exps.memory[inds]

                self.acmodel.reset_hiddens()

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]

                    # Compute loss

                    model_results = self.acmodel(sb.obs, memory * sb.mask, mask=sb.mask)
                    dist = model_results["dist"]
                    value = model_results["value"]
                    memory = model_results["memory"]
                    extra_predictions = model_results["extra_predictions"]

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = (
                        torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                        * sb.advantage
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(
                        value - sb.value, -self.clip_eps, self.clip_eps
                    )
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = (
                        policy_loss
                        - self.entropy_coef * entropy
                        + self.value_loss_coef * value_loss
                    )

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                # because we have a loop on variables in NPS and shared layers occur, set retain_graph to True
                batch_loss.backward()
                grad_norm = (
                    sum(
                        p.grad.data.norm(2) ** 2
                        for p in self.acmodel.parameters()
                        if p.grad is not None
                    )
                    ** 0.5
                )
                torch.nn.utils.clip_grad_norm_(
                    self.acmodel.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Update log values
                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm.item())
                log_losses.append(batch_loss.item())

        # Log some values

        logs["entropy"] = numpy.mean(log_entropies)
        logs["value"] = numpy.mean(log_values)
        logs["policy_loss"] = numpy.mean(log_policy_losses)
        logs["value_loss"] = numpy.mean(log_value_losses)
        logs["grad_norm"] = numpy.mean(log_grad_norms)
        logs["loss"] = numpy.mean(log_losses)
        logs["x_clip_loss"] = numpy.mean(x_clip_losses)
        

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch

        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [
            indexes[i : i + num_indexes] for i in range(0, len(indexes), num_indexes)
        ]

        return batches_starting_indexes
    
    def calculate_contrastive_loss(self, similarity_matrix):
        v_2_t_loss = 0
        t_2_v_loss = 0
        transposed_similarity_matrix = similarity_matrix.T
        log_softmax_row_wise = F.log_softmax(similarity_matrix, dim=1)
        log_softmax_column_wise = F.log_softmax(transposed_similarity_matrix, dim=1)

        v_2_t_loss = torch.trace(log_softmax_row_wise)

        v_2_t_loss = v_2_t_loss / -log_softmax_row_wise.shape[0]

        t_2_v_loss = torch.trace(log_softmax_column_wise)

        t_2_v_loss = t_2_v_loss / -log_softmax_column_wise.shape[0]

        total_loss = v_2_t_loss + t_2_v_loss

        return total_loss

    def Attention_Over_Similarity_Vector(self, vector, temp=1):
        vector_tmp = vector / temp
        attn_weights = F.softmax(vector_tmp, dim=0)
        weighted_sum = torch.dot(attn_weights, vector)
        return weighted_sum

    def Attention_Over_Similarity_Matrix(self, matrix, temp=1):
        matrix_tmp = matrix / temp
        attn_col_weights = F.softmax(matrix_tmp, dim=0)
        col_product = torch.mul(attn_col_weights, matrix)
        col_sum = torch.sum(col_product, dim=0)
        weighted_col_sum = self.Attention_Over_Similarity_Vector(col_sum, temp)

        attn_row_weights = F.softmax(matrix_tmp, dim=1)
        row_product = torch.mul(attn_row_weights, matrix)
        row_sum = torch.sum(row_product, dim=1).reshape(-1)
        weighted_row_sum = self.Attention_Over_Similarity_Vector(row_sum, temp)

        return (weighted_col_sum + weighted_row_sum) / 2

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
        # print(f'video_frames_shape: {video_frames.shape}')
        # print(f'positional_encodings_shape: {self.positional_encodings[:, :video_frames.size(1)].shape}')
        video_frames = video_frames + self.positional_encodings[:, :video_frames.size(1)].to(self.device)
        video_frames = video_frames.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        output, _ = self.multihead_attention(video_frames, video_frames, video_frames)
        output = output.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
        average_embedding = torch.mean(output, dim=1)  # (batch_size, embed_dim)
        return average_embedding

    def _generate_positional_encodings(self, max_length, embed_dim):
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        positional_encodings = torch.zeros(1, max_length, embed_dim)
        positional_encodings[:, :, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, :, 1::2] = torch.cos(position * div_term)
        return positional_encodings