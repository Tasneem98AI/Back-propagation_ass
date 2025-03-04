#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

inputs = np.array([0.05, 0.10])  

weights = {
    "w1": 0.15, "w2": 0.20, "w3": 0.25, "w4": 0.30,
    "w5": 0.40, "w6": 0.45, "w7": 0.50, "w8": 0.55
}

biases = {"b1": 0.35, "b2": 0.60}

targets = np.array([0.01, 0.99])  

learning_rate = 0.5


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


# In[2]:


h1_input = inputs[0] * weights["w1"] + inputs[1] * weights["w3"] + biases["b1"]
h2_input = inputs[0] * weights["w2"] + inputs[1] * weights["w4"] + biases["b1"]

h1_output = sigmoid(h1_input)
h2_output = sigmoid(h2_input)

o1_input = h1_output * weights["w5"] + h2_output * weights["w7"] + biases["b2"]
o2_input = h1_output * weights["w6"] + h2_output * weights["w8"] + biases["b2"]

o1_output = sigmoid(o1_input)
o2_output = sigmoid(o2_input)


# In[3]:


error_o1 = 0.5 * (targets[0] - o1_output) ** 2
error_o2 = 0.5 * (targets[1] - o2_output) ** 2
total_error = error_o1 + error_o2


# In[4]:


d_o1 = (o1_output - targets[0]) * sigmoid_derivative(o1_output)
d_o2 = (o2_output - targets[1]) * sigmoid_derivative(o2_output)

weights["w5"] -= learning_rate * d_o1 * h1_output
weights["w6"] -= learning_rate * d_o2 * h1_output
weights["w7"] -= learning_rate * d_o1 * h2_output
weights["w8"] -= learning_rate * d_o2 * h2_output

d_h1 = (d_o1 * weights["w5"] + d_o2 * weights["w6"]) * sigmoid_derivative(h1_output)
d_h2 = (d_o1 * weights["w7"] + d_o2 * weights["w8"]) * sigmoid_derivative(h2_output)


# In[5]:


weights["w1"] -= learning_rate * d_h1 * inputs[0]
weights["w2"] -= learning_rate * d_h1 * inputs[1]
weights["w3"] -= learning_rate * d_h2 * inputs[0]
weights["w4"] -= learning_rate * d_h2 * inputs[1]

print("Updated Weights:")
for key, value in weights.items():
    print(f"{key}: {value:.4f}")


# In[ ]:




