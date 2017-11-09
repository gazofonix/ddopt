
# coding: utf-8

# # Simulator for NUM problems

# In[4]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# Function definition

# In[5]:


def generate_link(link):
    """ Generate random link capacity """

    scale = 1000
    return scale*np.random.rand(link)


# In[6]:


def generate_path(source, path, link):
    """ Generate random paths """
    
    x = np.zeros((source, np.max(path), link))
    for i in range(source):
        x[i,:,:] = np.round(np.random.rand(path[i], link))

    return x


# In[7]:


def compute_utility():
    """ Compute the utility at the current iteration """
    
    return sum(log(sum(max(x, [], 2))))


# In[8]:


def compute_step():
    """ Compute the step size at the current iteration """
    
    return


# In[9]:


def gradient_step():
    """ Compute the gradient step at the current iteration """
    
    return


# Variable initialization

# In[10]:


# Max number of iterations
max_iter =  1000

# Number of OD pairs
source = 5

# Number of paths per OD pair
path = 5*np.ones(source, dtype='int32')

# Number of links
link = 10


# In[13]:


# Generate link capacity
cl = generate_link(link)
print(cl)

# Generate paths
x = generate_path(source, path, link)
print(x)


# In[ ]:


utility = np.zeros((max_iter, 1))

for i in range(1, max_iter):
    
    utility[i] = (compute_utility())
    
    for j in range(1, source):
    
        # Gradient step
        step_size = compute_step()
        x[:, :, j] = gradient_step()
        
print (utility)

