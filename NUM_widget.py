
# coding: utf-8

# In[ ]:


from nbparameterise import extract_parameters, parameter_values, replace_definitions
import nbformat
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display, clear_output


# In[ ]:


def update_input(a, b, c, d):
    
    # Open notebook
    with open("Network Utility Maximization.ipynb") as f:
        nb = nbformat.read(f, as_version=4)
    
    # Update the parameters
    orig_parameters = extract_parameters(nb)
    params = parameter_values(orig_parameters, max_iter=a, source=b, max_path=c, link=d)
    new_nb = replace_definitions(nb, params, False)
    
    # Save changes
    with open("Network Utility Maximization.ipynb", 'w') as f:
        nbformat.write(new_nb, f)


# In[ ]:


def run_simulator():

    # Run notebook
    get_ipython().magic('run "Network Utility Maximization.ipynb"')


# In[ ]:


def f1(a1):
    
    global max_iter
    max_iter = a1


# In[ ]:


def f2(a2):
    
    global source
    source = a2


# In[ ]:


def f3(a3):
    
    global max_path
    max_path = a3


# In[ ]:


def f4(a4):
    
    global link
    link = a4


# In[ ]:


# Create sliders
w1 = widgets.IntSlider(description="Iterations", min=100, max=10000, step=100, value=1000)
interact(f1, a1=w1)
w2 = widgets.IntSlider(description="Sources", min=1, max=1000, step=1, value=100)
interact(f2, a2=w2)
w3 = widgets.IntSlider(description="Paths", min=1, max=1000, step=1, value=100)
interact(f3, a3=w3)
w4 = widgets.IntSlider(description="Links", min=1, max=1000, step=1, value=100)
interact(f4, a4=w4)

# Create command button
button = widgets.Button(description="Run simulator")
display(button)
    
def on_button_clicked(b):
    clear_output()
    print("Iterations:", max_iter)
    print("Sources:", source)
    print("Paths:", max_path)
    print("Links:", link)
    print("Updating input variables...")
    update_input(max_iter, source, max_path, link)
    print("Variables updated successfully")
    print("Simulator starts")
    run_simulator()
    
button.on_click(on_button_clicked)

