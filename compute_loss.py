import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class ContrastiveLoss:
    def __init__(self, temperature=0.07):
        """
        Initialize the ContrastiveLoss class.

        Args:
            temperature (float): Temperature parameter for scaling the logits.
        """
        self.temperature = temperature

    def __call__(self, feature1,feature2):
        """
        Compute the contrastive loss for a batch of sentence embeddings.

        Args:
            embeddings (torch.Tensor): A tensor of shape (2N, D) containing embeddings for the mini-batch,
                                       where N is the batch size and D is the embedding dimension. The first N embeddings
                                       are the original sentences and the second N embeddings are the augmented (dropout) versions.

        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        return self.compute_loss( feature1,feature2 )

    def compute_loss(self, feature1, feature2):
        """
        Compute the contrastive loss.

        Args:
            feature1 (torch.Tensor): A tensor of shape (N, D) containing embeddings for the first mini-batch.
            feature2 (torch.Tensor): A tensor of shape (N, D) containing embeddings for the second mini-batch.

        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        # Normalize embeddings
        feature1_norm = feature1 / feature1.norm(dim=1, keepdim=True)
        feature2_norm = feature2 / feature2.norm(dim=1, keepdim=True)

        # Compute cosine similarity
        similarity_matrix = torch.matmul(feature1_norm, feature2_norm.T)

        # Create labels
        batch_size = feature1.shape[0]
        labels = torch.arange(batch_size).to(feature1.device)

        # Mask to ignore self-similarity (not needed here since we have two separate batches)
        # mask = torch.eye(batch_size, dtype=torch.bool).to(feature1.device)

        # Compute logits
        logits = similarity_matrix / self.temperature
        # Apply mask to the logits to ignore self-similarity (if applicable)
        # logits.masked_fill_(mask, float('-inf'))

        # Compute cross-entropy loss
        loss1 = F.cross_entropy(logits, labels)
        loss2 = F.cross_entropy(logits.T, labels)

        loss = 0.5 * loss1 + 0.5 * loss2

        return loss

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.2, contrast_mode='all', base_temperature=0.2):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, features):

        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        features = F.normalize(features, dim=-1)
        batch_size = features.shape[0]

        mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            self.leaky_relu(torch.matmul(anchor_feature, contrast_feature.T)),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

