
# coding: utf-8

# # Simulator for NUM problems

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import pdb


# Function definition

# In[13]:


def generate_link(link):
    """ Generate random link capacity """

    scale = 1000
    return scale * np.random.rand(link)


# In[14]:


def generate_path(source, path, link):
    """ Generate random paths """
    
    x = np.zeros((source, np.max(path), link))
    for i in range(source):
        x[i,:,:] = np.round(np.random.rand(path[i], link))

    return x


# We set the initial solution as
# \begin{equation}
#     x^{(0)}_{s,p}=\min_{l\in L}\left\{ \frac{0,9\cdot c_l}{\sum_{s,p}\mathbf{1}_{s,p\ni l}}\right\}, \qquad \forall s\in S, p\in P_s
# \end{equation}

# In[15]:


def initial_solution(x, cl):
    """ Compute an initial feasible solution """
    
    return x * np.min((0.9 * cl)/ np.sum(np.sum(x, axis=1), axis=0))


# We use the following utility function:
# \begin{equation}
#     \sum_{s\in S}a_s\cdot\log\left(\sum_{p\in P_s}x_{s,p}\right), \qquad a_s\in [0, 1].
# \end{equation}

# In[16]:


def compute_utility(x, source):
    """ Compute the utility at the current iteration """
    
    return sum(np.random.rand(source) * np.log(np.sum(np.max(x, axis=2), axis=1)))


# In[17]:


def backtracking_stepsize():
    """ Compute backtracking line search for the step size at the current iteration """
    
    return


# In[18]:


def plot_utility(y):
    """ Plot utility value per iteration """
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(np.shape(y)[0]), y)
    plt.tight_layout()
    ax.set_title('Utility')
    plt.show()


# We use the barrier function:

# In[19]:


def gradient_step(old_x, source, link, path, step_size, barrier, cl):
    """ Compute the gradient step at the current iteration """
    
    # We approximate infinity with a large value
    INF = np.power(10, 10)
    
    # Compute price
    price = np.zeros((source, np.max(path)))
    link_price = 1 / (cl - np.sum(np.sum(old_x, axis=1), axis=0))
    link_price[link_price == np.inf] = INF 
    for s in range(source):
        for p in range(path[s]):
            price[s, p] = np.dot((old_x[s, p, :] > 0).astype(int), link_price)

    # Gradient step
    x = np.max(old_x, axis=2) * np.exp(step_size * (1 / np.max(old_x, axis=2) - price))
    return np.tile(x, (link, 1, 1)).transpose(1, 2, 0) * (old_x > 0).astype(int)


# Variable initialization

# In[20]:


# Max number of iterations
max_iter =  10000

# Number of OD pairs
source = 5

# Number of paths per OD pair
path = 5 * np.ones(source, dtype='int32')

# Number of links
link = 10

# Fixed step size
step_size = 0.01
barrier = 0.01

# Output array of utilities
utility = np.zeros((max_iter + 1))


# In[21]:


# Generate link capacity
cl = generate_link(link)

# Generate paths
x = generate_path(source, path, link)


# Body of the simulator

# In[22]:


# Initial feasible solution
x = initial_solution(x, cl)

utility[0] = (compute_utility(x, source))
for i in range(max_iter):
#    step_size = backtracking_stepsize()    Uncomment to use backtracking line search
    x = gradient_step(x, source, link, path, step_size, barrier, cl)
    utility[i] = (compute_utility(x, source))
        
# Plot utility
plot_utility(utility)

