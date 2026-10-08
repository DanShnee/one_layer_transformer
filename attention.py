"""The four-head attention stage of the forward pass."""

import torch


def attention(self, x0: torch.Tensor) -> torch.Tensor:
    """Return x1 of shape [batch, context, model width] from embedded input 
    x0 of same shape.

    For each of the four heads h, compute Q=x0*W_Q[h], K=x0*W_K[h],
    V=x0*W_V[h], score matrix S=Q*K^t, and attention weights A=softmax(S). 
    Output of each head is A*V*W_O[h], sum the
    projected head outputs, and add the residual x0.     
    """
    Q = torch.einsum('bcd,hde->hbce',x0,self.W_Q)   #h is number of heads
    K = torch.einsum('bcd,hde->hbce',x0,self.W_K)   #b is batches, d is embedding dimension
    V = torch.einsum('bcd,hde->hbce',x0,self.W_V)   #c is context length, e is attention dimension

    S = Q @ K.transpose(-2,-1)  #Attention scores shape (hbcc) how the tokens in context relate to each other
    A = torch.softmax(S,dim = -1)   #softmax over a specific query position
    x1 = x0 + torch.einsum('hbci,hbie,hed->bcd',A,V,self.W_O)

    return x1
