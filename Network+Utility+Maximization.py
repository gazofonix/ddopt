
# coding: utf-8

# # Simulator for NUM problems

# In[120]:


max_iter = 1000
source = 10
link = 10
max_path = 10
barrier = 1


# In[121]:


import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle


# In[122]:


get_ipython().magic('matplotlib notebook')


# In[123]:


def generate_coeff(source):
    """ Generate random coefficients for the utility function """    
    
    return np.random.rand(source)


# In[124]:


def generate_link(link):
    """ Generate random link capacity """

    scale = 100
    return scale * np.random.rand(link)


# In[125]:


def generate_path(source, path, link):
    """ Generate random paths """
    
    x = np.zeros((source, np.max(path), link))
    for i in range(source):
        x[i,:,:] = np.round(np.random.rand(path[i], link))
        
    # check if some paths have zero links
    aux = np.where(np.sum(x, axis=2) == 0)[1]
    for idx, i in enumerate(np.where(np.sum(x, axis=2) == 0)[0]):
        z = int(np.floor(link * np.random.rand()))
        x[i,aux[idx], z] = 1
    
    return x


# In[126]:


def generate_graph():
    
    # Define utility
    coeff = generate_coeff(source)

    # Generate link capacity
    cl = generate_link(link)

    # Generate paths
    x0 = generate_path(source, path, link)
    
    data = {"coeff": coeff, "link_capacity": cl, "path": x0}
    
    # Store graph information in an external file 'graph.p'
    pickle.dump(data, open("graph.p", "wb"))
    
    return data


# We set the initial solution as
# \begin{equation}
#     x^{(0)}_{s,p}=\min_{l\in L}\left\{ \frac{0,9\cdot c_l}{\sum_{s,p}\mathbf{1}_{s,p\ni l}}\right\}, \qquad \forall s\in S, p\in P_s
# \end{equation}

# In[127]:


def initial_solution(x, cl):
    """ Compute an initial feasible solution according to the above expression """
    
    return x * np.min((0.9 * cl)/ np.sum(np.sum(x, axis=1), axis=0))


# We use the following utility function:
# \begin{equation}
#     \sum_{s\in S}a_s\cdot\log\left(\sum_{p\in P_s}x_{s,p}\right), \qquad a_s\in [0, 1].
# \end{equation}

# In[128]:


def compute_utility(x, source, coeff):
    """ Compute the utility at the current iteration """
    
    return sum(coeff * np.log(np.sum(np.max(x, axis=2), axis=1)))


# In[129]:


def compute_obj(x, source, coeff, b, cl):
    """ Compute the utility minus the barrier penalty """

    return compute_utility(x, source, coeff) - sum(b * np.log(cl - np.sum(np.sum(x, axis=1), axis=0)))


# In[130]:


def backtracking_linesearch(old_x, source, link, path, coeff, step_size, barrier, cl):
    """ Compute backtracking line search for the step size at the current iteration """
    
    a = 0.1    # Typical values are in the range [0.01, 0.3]
    b = 0.5    # Typical values are in the range [0.1, 0.8]
    t = 1      # Initial step size
    
    x = gradient_step(old_x, source, link, path, t, barrier, cl)
    while compute_obj(x, source, coeff, barrier, cl) - compute_obj(old_x, source, coeff, barrier, cl) < a * t:
        t = b * t
        x = gradient_step(old_x, source, link, path, t, barrier, cl)
    
    return t


# In[131]:


def load_graph(a):

    return a["coeff"], a["link_capacity"], a["path"]


# We use the logarithmic barrier function
# \begin{equation}
#     \quad B_l({y}) = \begin{cases} -\log(c_l - y), & \mbox{if } {y}< c_l, \\ \infty, & \mbox{if } {y}\geq c_l. \end{cases}
# \end{equation}
# 
# The gradient step for the exponentiated gradient descent is
# \begin{equation}
#     x_{s,p}^{(k+1)} = x_{s,p}^{(k)}\cdot \exp \left\{\eta^{(k)}\cdot \left[U_s'\left(x^{(k)}_{s,p}\right) - \sum_{l:l\in s,p}\frac{\mu^{(k)}}{\sum x^{(k)}_{s,p} - c_l}\right]\right\}
# \end{equation}

# In[132]:


def egd_step(old_x, source, link, path, step_size, barrier, cl):
    """ Compute the gradient step (EGD algorithm) at the current iteration """
    
    price = compute_price(source, path, cl, old_x)

    # Gradient step
    aux = np.max(old_x, axis=2) * np.exp(step_size * (1 / np.max(old_x, axis=2) - price))
    x = np.tile(aux, (link, 1, 1)).transpose(1, 2, 0) * (old_x > 0).astype(int)
    while any(np.sum(np.sum(x, axis=1), axis=0) > cl):
        step_size *= 0.9
        aux = np.max(old_x, axis=2) * np.exp(step_size * (1 / np.max(old_x, axis=2) - price))
        x = np.tile(aux, (link, 1, 1)).transpose(1, 2, 0) * (old_x > 0).astype(int)
        
    return x, step_size


# In[133]:


def gd_step(old_x, source, link, path, step_size, barrier, cl):
    """ Compute the gradient step (GD algorithm) at the current iteration """
    
    price = compute_price(source, path, cl, old_x)

    # Gradient step
    aux = np.max(old_x, axis=2) + step_size * (1 / np.max(old_x, axis=2) - price)
    x = np.tile(aux, (link, 1, 1)).transpose(1, 2, 0) * (old_x > 0).astype(int)
    while any(np.sum(np.sum(x, axis=1), axis=0) > cl) or (np.sum(np.sum(aux < 0)) > 0):
        step_size *= 0.9
        aux = np.max(old_x, axis=2) + step_size * (1 / np.max(old_x, axis=2) - price)
        x = np.tile(aux, (link, 1, 1)).transpose(1, 2, 0) * (old_x > 0).astype(int)
        
    return x, step_size 


# In[134]:


def compute_price(source, path, cl, x):
    """ Compute the 'price' intriduced by the barrier function the current iteration """
    
    price = np.zeros((source, np.max(path)))
    link_price = 1 / (cl - np.sum(np.sum(x, axis=1), axis=0))
    for s in range(source):
        for p in range(path[s]):
            price[s, p] = np.dot((x[s, p, :] > 0).astype(int), link_price)
            
    return price


# In[135]:


def check_overflow(x, step_size):
    """ Bound the exponent of the gradient step with the value set by <overflow_exponent> """
    
    overflow_exponent = 15
    if step_size * (1 / x) > overflow_exponent:
        step_size = np.min(overflow_exponent / (1 / x))

    return step_size


# In[136]:


def exponentiated_gradient_descent(x0, cl, max_iter, source, link, path, coeff, barrier):
    """ Compute the utility of the EGD algorithm """
    
    step_correction = 0.9999 # decrease step size
    
    # Initial feasible solution EGD
    x = initial_solution(x0, cl)
    step_size = check_overflow(np.min(np.max(x, axis=2)), 10)
    utility = np.zeros((max_iter + 1))
    utility[0] = compute_utility(x, source, coeff)
    
    # Output variables
    obj = np.zeros((max_iter + 1))
    obj[0] = compute_obj(x, source, coeff, step_size, cl)
    step = np.zeros((max_iter + 1))
    step[0] = step_size

    for i in range(max_iter):
        step_size *= step_correction
        x, step_size = egd_step(x, source, link, path, step_size, step_size, cl)
            
        # Output
        utility[i + 1] = compute_utility(x, source, coeff)
        obj[i + 1] = compute_obj(x, source, coeff, step_size, cl)
        step[i + 1] = step_size
        
        if i % np.round(max_iter / 10) == 0:
            print(i)

    return utility, obj, step


# In[137]:


def gradient_descent(x0, cl, max_iter, source, link, path, coeff, barrier):
    """ Compute the utility of the GD algorithm """

    step_correction = 0.9999 # decrease step size
    
    # Initial feasible solution GD
    x = initial_solution(x0, cl)
    step_size = 100
    utility = np.zeros((max_iter + 1))
    utility[0] = compute_utility(x, source, coeff)
    
    # Output variables
    obj = np.zeros((max_iter + 1))
    obj[0] = compute_obj(x, source, coeff, step_size, cl)
    step = np.zeros((max_iter + 1))
    step[0] = 0


    for i in range(max_iter):
        step_size *= step_correction
        x, step_size = gd_step(x, source, link, path, step_size, barrier, cl)
            
        # Output
        utility[i + 1] = compute_utility(x, source, coeff)
        obj[i + 1] = compute_obj(x, source, coeff, step_size, cl)
        step[i + 1] = step_size
        
        if i % np.round(max_iter / 10) == 0:
            print(i)

    return utility, obj, step


# In[138]:


def generate_plot(utility_egd, utility_gd, max_iter):
    """ Plot utility value per iteration """
    
    fig, ax = plt.subplots()
    ax.plot(range(max_iter), utility_egd[1:], 'k', label='EGD')
    ax.plot(range(max_iter), utility_gd[1:], 'g:', label='GD')
    
    # Add the legent
    legend = ax.legend(loc='lower right', shadow=True)

    # The frame is matplotlib.patches.Rectangle instance surrounding the legend
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.2)
    
    plt.show()


# Body of the simulator

# In[139]:


# Number of paths per OD pair
path = max_path * np.ones(source, dtype='int32')

# Upload graph information
try:
    b = pickle.load(open("graph.p", "rb"))  
    coeff, cl, x0 = load_graph(b)
except:
    print("Generating a new network graph...")
    data = generate_graph()
    coeff, cl, x0 = load_graph(data)
    print("Network graph generated!")

# Compute utility
get_ipython().magic('time utility_egd, obj_egd, step_egd = exponentiated_gradient_descent(x0, cl, max_iter, source, link, path, coeff, barrier)')
get_ipython().magic('time utility_gd, obj_gd, step_gd = gradient_descent(x0, cl, max_iter, source, link, path, coeff, barrier)')
       
# Plot
generate_plot(utility_egd, utility_gd, max_iter)
generate_plot(obj_egd, obj_gd, max_iter)
generate_plot(step_egd, step_gd, max_iter)

