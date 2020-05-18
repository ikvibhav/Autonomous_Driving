#Numpy is a great framework, but it cannot utilize GPUs to accelerate its 
#numerical computations. For modern deep neural networks, GPUs often provide 
#speedups of 50x or greater, so unfortunately numpy won’t be enough for 
#modern deep learning.

# -*- coding: utf-8 -*-

import torch


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)	#64*1000
y = torch.randn(N, D_out, device=device, dtype=dtype)	#64*10

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)	#1000*100
w2 = torch.randn(H, D_out, device=device, dtype=dtype)	#100*10

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)										#64*100
    h_relu = h.clamp(min=0)								#64*100
    y_pred = h_relu.mm(w2)								#64*10

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()			 	#.sum() with no specified dimension outputs a single value
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)					#64*10
    grad_w2 = h_relu.t().mm(grad_y_pred)				#(64*100)^T * (64*10) = 100*10
    grad_h_relu = grad_y_pred.mm(w2.t())				#(64*10) * (100*10)^T = 64*100
    grad_h = grad_h_relu.clone()						#64*100
    grad_h[h < 0] = 0									#Why do we make -ves 0?
    grad_w1 = x.t().mm(grad_h)							#(64*1000)^T * 64*100

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2