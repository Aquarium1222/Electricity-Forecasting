import random

import numpy as np
import torch
import torch.nn as nn

from config import *


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()

        self.feature_dim = Hyperparameter.FEATURE_DIM
        self.output_dim = Hyperparameter.OUTPUT_DIM
        self.output_len = Hyperparameter.OUTPUT_SEQ_LEN
        self.teacher_forcing_ratio = Hyperparameter.TEACHER_FORCING_RATIO

        self.encoder = encoder
        self.linear = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Linear(self.feature_dim, self.output_dim)
        )
        self.decoder = decoder

    def forward(self, x, y, teacher_forcing_ratio=Hyperparameter.TEACHER_FORCING_RATIO):
        """
        param x: [input_seq_len, batch_size, feature_dim]
        param y: [output_seq_len, batch_size, feature_dim]
        """
        h, c = self.encoder(x)
        decoder_input = self.linear(x[-1, :, :].unsqueeze(0))
        outputs = torch.zeros(y.shape)
        for i in range(self.output_len):
            output, h, c = self.decoder(decoder_input, h, c)
            outputs[i] = output
            is_teacher_forcing = random.random() < teacher_forcing_ratio
            decoder_input = y[i] if is_teacher_forcing else output
            if len(decoder_input.shape) != 3:
                decoder_input = decoder_input.unsqueeze(0)
        return outputs
