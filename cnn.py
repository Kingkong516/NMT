#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1h
class CNN(nn.Module):

    def __init__(self,input_size,l_in,output_size,kernel_size=5):
        """Init the convolution layer.

        @param input_size/channel (int): char embedding size (dimensionality)
        @param l_in (int): max length of a word (dimensionality)
        @param output_size/channel (int): word embedding size (dimensionality)
        @param k (int): kernel size
        """
        super(CNN,self).__init__()
        self.conv = nn.Conv1d(input_size,output_size,kernel_size=kernel_size)
        self.maxpool = nn.MaxPool1d(l_in-kernel_size+1)
        
    def forward(self, x):
        """ Take a batch of sentences with embeded characters, use a convolutional 
        layer and a maxpooling layer to transform to word embeddings.

        @param x (Tensor): a variable/tensor of shape (b, sent_len, m_word, e_char), output of char embedding.

        @returns x_conv_out (Tensor): a variable/tensor of shape (b, sent_len, e_word).
        """
        shape = x.shape
        high_dim = len(shape)>3
        if high_dim:
            x = x.view([-1]+list(shape[-2:]))
        x_conv = F.relu(self.conv(x))
        x_conv_out = self.maxpool(x_conv)
        if high_dim:
            x_conv_out = x_conv_out.view(list(shape[:-2])+[-1])
        return x_conv_out
        
### END YOUR CODE

