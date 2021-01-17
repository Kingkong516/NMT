#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    
    def __init__(self,input_size,output_size,dropout_rate=0.5):
        """Init the Highway layers.

        @param input_size (int): embedding size (dimensionality) 
        @param output_size (int): embedding size (dimensionality)
        @param dropout_rate (float): dropout rate of the dropout layer
        """
        super(Highway,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.proj = nn.Linear(self.input_size, self.output_size, bias=True)
        self.gate = nn.Linear(self.input_size, self.output_size, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_rate, inplace=False)
    
    def forward(self, x):
        """ Take a batch of sentences with embeded words, use a highway 
        layer to transform the word embeddings.

        @param x (Tensor): a variable/tensor of shape (b, sent_len, e_word), output of conv layer.

        @returns x_word_embed (Tensor): a variable/tensor of shape (b, sent_len, e_word).
        """
        x_proj = F.relu(self.proj(x))
        x_gate = self.sigmoid(self.gate(x))
        x_highway = x_gate*x_proj+(1-x_gate)*x
        x_word_emd = self.dropout(x_highway)
        return x_word_emd

### END YOUR CODE 

