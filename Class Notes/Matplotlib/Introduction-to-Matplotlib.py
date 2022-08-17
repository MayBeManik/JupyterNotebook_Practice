#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


x=[1,2,3,4]
y=[11,22,33,45]
plt.plot(x,y);


# In[3]:


# Recommended Method
fig , ax= plt.subplots()
ax.plot(x,[50,100,200,250]);


# In[4]:


plt.plot(x,y)


# In[5]:


fig , ax= plt.subplots(figsize=(5,5)) #figsize = set the size of plot(width, height)
ax.plot(x,y,"ro--")
ax.set(title="Simple plot",
       xlabel="x-Axis",
      ylabel="y-Axis");


# # Steps for a Plot

# In[6]:


# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 2. Prepare data
x=[1,2,3,4]
y=[11,22,33,45]

# 3. setup plot
fig , ax= plt.subplots(figsize=(5,5))

# 4. Plot data
ax.plot(x,y,"ro--")

# 5. Customize plot
ax.set(title="Simple plot",
       xlabel="x-Axis",
      ylabel="y-Axis");


# 6. Save & show the plot
#fig.save("images/sample-plot.png")


# In[7]:


sugar_men = [10,12,22,34,54,67,77,82,34,34,57,20,30,40,50]
sugar_women = [15,25,35,45,55,34,54,65,45,45,35,43,76,54,32]
fig, ax=plt.subplots()
ax.plot(sugar_men,sugar_women)
ax.set(title="sugar range",
       xlabel="Men",
       ylabel="Women")  ;   # ; help to remove extra lines from the code


# # Making figures with NumPy arrays
# 
# We want:
# * Line plot
# * Sactter plot
# * Bar plot
# * Hitogram
# * Subplots

# In[8]:


import numpy as np


# In[9]:


x = np.linspace(1,10,100)
x


# In[10]:


# Line Plot
fig, ax=plt.subplots()
ax.plot(x, x**2)


# In[11]:


# Scattter Plot
fig, ax = plt.subplots()
ax.scatter(x ,np.exp(x));


# In[12]:


# Make a plot from dictionary 
# Bar Plot
anime_sales = { "Demon Slayer" : 10,
               "Hunter X Hunter" : 15,
               "Jujutsu Kaisen" : 25,
               "Death Note" : 20
              }
fig, ax= plt.subplots()
ax.bar(anime_sales.keys(), anime_sales.values())
ax.set(title="Amine Sales",
      xlabel="Anime Names",
      ylabel="No. of Sales");


# In[13]:


# Bar Plot(Another way , Data same as above)
plt.title("Amine Sales")
plt.xlabel("Anime Names")
plt.ylabel("No. of Sales")
plt.bar(anime_sales.keys(), anime_sales.values(), width =0.5 ,color ="r", edgecolor="g",linewidth=5)


# In[14]:


# Histogram

# Create a data
x = np.random.randn(1000)

# Now plot it
fig,ax = plt.subplots()
ax.hist(x,rwidth=0.95);


# In[15]:


# Another way of plotting Histogram

# Create a data
x = np.random.randn(1000)

# Now Plot
plt.hist(x,rwidth=.95);


# ### Two options for subplots

# In[16]:


# Subplot Option 1
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,
                                         ncols=2,
                                         figsize=(10,5))

# Plot wiht tuples like ((ax1,ax2),(ax3,ax4))

# Fig 1
ax1.plot(x,x/2)

# Fig 2
ax2.scatter(np.random.random(10),np.random.random(10),color="r")

# Fig 3
ax3.bar(anime_sales.keys(),anime_sales.values(),width=0.5)

# Fig 4
ax4.hist(np.random.randn(1010),rwidth=.95);


# In[17]:


# Subplot Option 2
fig, ax=plt.subplots(nrows=2,
                    ncols=2,
                    figsize=(10,5))

# Plot to each diffrent index

# Fig 1
ax[0,0].plot(x,x/2)

# Fig 2
ax[0,1].scatter(np.random.random(10),np.random.random(10),color="r")

# Fig 3
ax[1,0].bar(anime_sales.keys(),anime_sales.values(),color="g",width=.5)

# Fig 4
ax[1,1].hist(np.random.randn(1000),rwidth=.95);


# ## Plotting from Pandas DataFrame

# In[18]:


import pandas as pd


# In[19]:


# Example series data
ts = pd.Series(np.random.randn(1000),
               index=pd.date_range("1/1/2022",periods=1000))
ts = ts.cumsum()
ts.plot();


# In[20]:


# Make a DataFrame
car_sales = pd.read_csv("car_sales.csv")
car_sales


# In[21]:


# Removing symbols from Price Column
car_sales["Price"]=car_sales["Price"].str.replace('[\$\,\.]','')
car_sales


# In[22]:


# Removing Extra Zeros from the Price Column
car_sales["Price"] = car_sales["Price"].str[:-2]
car_sales


# In[23]:


# Adding a Date Column
car_sales["Sales Date"] = pd.date_range("1/1/2022",periods=len(car_sales))
car_sales


# In[24]:


# Converting Price Column into Int Type
car_sales["Price"] = car_sales["Price"].astype(int)
car_sales


# In[25]:


# Creating a New Column with Cumsum function
car_sales["Total Sales"] = car_sales["Price"].cumsum()
car_sales


# In[26]:


# Now Plotting Bar Graphs
car_sales.plot(x='Price',y='Total Sales',kind="bar", color="g")


# In[27]:


# Now Plotting Scatter Plot
car_sales.plot(x='Total Sales', y='Odometer (KM)',kind="scatter")


# In[28]:


car_sales.plot(x = "Price",kind="bar",)


# In[29]:


# Now Plotting Histogram Plot
car_sales.plot(x="Price",y="Odometer (KM)",kind="hist")


# # New DataFrame

# In[30]:


heart = pd.read_csv("heart-disease.csv")
heart


# ## Which One Should we use? (Pyplot vs Matplot OO Method)
# .When Plotting some thing quickly, use Pyplot Method
# 
# .When plotting something more advanced, use the OO method

# In[31]:


heart


# In[32]:


# Creating new table for over 50 age in heart DataFrame
over_50 = heart[heart["age"]>50]
over_50


# In[33]:


# Pyplot Method
over_50.plot(x='age',
            y='chol',
            kind='scatter',
            figsize=(10,5),
            c='target'); # c means color from target coloumn


# In[34]:


# OO Method
fig , ax=plt.subplots(figsize=(10,5))
over_50.plot(x='age',
            y='chol',
            c='target',
             kind='scatter',
            ax=ax)

ax.set_xlim([45,100]); # It set the limit of x axis from 45 to 100


# In[35]:


# OO method from Scratch
fig, ax = plt.subplots(figsize=(10,5))

# Plot tha Data
scatter=ax.scatter(x=over_50['age'],
           y=over_50['chol'],
           c=over_50['target'])

# Customize the Plot
ax.set(title="Heart Disease and Cholesterol Level",
       xlabel="Age",
       ylabel="Cholesterol")

# Add Legend
ax.legend(*scatter.legend_elements(),title='Target')

# Add as Horizontal Line
ax.axhline(over_50["chol"].mean(),linestyle="--");


# In[36]:


# SubPlots
fig,(ax0,ax1)=plt.subplots(figsize=(10,10),
                            nrows=2,
                            ncols=1,
                            sharex=True) #sharex let both th polt to have same xlable

# Add data to ax0
scatter = ax0.scatter(x=over_50["age"],
                      y=over_50['chol'],
                      c=over_50["target"])

# Customize the ax0 plot
ax0.set(title="Heart Disease and Cholesterol Level",
       ylabel="Cholesterol")

# Mean Line in ax0 plot
ax0.axhline(over_50["chol"].mean(), linestyle="--")

# Add legends in ax0 plot
ax0.legend(*scatter.legend_elements(),title="Target")


# Add data to ax1
scatter = ax1.scatter(x=over_50["age"],
                      y=over_50["thalach"],
                      c=over_50['target'])

# Customize the ax1 plot
ax1.set(title="Heart Disease and Thalach Level",
       xlabel="Age",
       ylabel="Max Heart rate")

# Add legend
ax1.legend(*scatter.legend_elements(), title="Target")

# Add Mean line
ax1.axhline(over_50["thalach"].mean(),linestyle="-.")

# Add a title to the figure
fig.suptitle("Heart Disease Analysis", fontsize=16, fontweight='bold');


# ## Customizing Mtaplotlib Plots

# In[37]:


# Diffrent types of styles available
plt.style.available


# In[38]:


car_sales["Price"].plot();


# In[39]:


plt.style.use("seaborn-whitegrid")


# In[40]:


car_sales["Price"].plot();


# In[41]:


plt.style.use('seaborn-darkgrid')


# In[42]:


car_sales["Price"].plot();


# In[43]:


# plt.style.use("grayscale")


# In[44]:


# car_sales["Price"].plot();


# In[45]:


# plt.style.use("seaborn-poster")


# In[46]:


# car_sales["Price"].plot();


# In[47]:


# plt.style.use("seaborn-talk")


# In[48]:


# car_sales["Price"].plot();


# In[49]:


plt.style.use("seaborn-notebook")


# In[50]:


# car_sales["Price"].plot()


# In[51]:


x=np.random.randn(10,4)
x


# In[52]:


df=pd.DataFrame(x,columns=["a",'b','c','d'])
df


# In[53]:


df.plot(kind='bar')


# In[54]:


# We will create above graph with more details

# Setting the style
plt.style.use("seaborn-whitegrid")

# creating the plot
ax=df.plot(kind='bar')

# Customizing th plot
ax.set(title="Random Number Bar Graph",
      xlabel="Row Number",
      ylabel="Random number")

# Adding Legends
ax.legend().set_visible(True)


# In[55]:


# Set Style
plt.style.use("seaborn-whitegrid")

# OO method from Scratch
fig, ax = plt.subplots(figsize=(10,5))

# Plot tha Data
scatter=ax.scatter(x=over_50['age'],
           y=over_50['chol'],
           c=over_50['target'],
                  cmap='winter')  #cmap changes the colour scheme

# Customize the Plot
ax.set(title="Heart Disease and Cholesterol Level",
       xlabel="Age",
       ylabel="Cholesterol")

# Add Legend
ax.legend(*scatter.legend_elements(),title='Target')

# Add as Horizontal Line
ax.axhline(over_50["chol"].mean(),linestyle="--",color='r');


# # This is the final Plot that show Heart Disease

# In[56]:


# SubPlots
fig,(ax0,ax1)=plt.subplots(figsize=(10,10),
                            nrows=2,
                            ncols=1,
                            sharex=True) #sharex let both the plot to have same xlable

# Add data to ax0
scatter = ax0.scatter(x=over_50["age"],
                      y=over_50['chol'],
                      c=over_50["target"],
                     cmap='winter')

# Customize the ax0 plot
ax0.set(title="Heart Disease and Cholesterol Level",
       ylabel="Cholesterol")

# Setting x axis limits (To remove extra spaces from tha x axis)
ax0.set_xlim([50,80])

# Mean Line in ax0 plot
ax0.axhline(over_50["chol"].mean(), linestyle="--",color='r')

# Add legends in ax0 plot
ax0.legend(*scatter.legend_elements(),title="Target")


# Add data to ax1
scatter = ax1.scatter(x=over_50["age"],
                      y=over_50["thalach"],
                      c=over_50['target'],
                     cmap='winter')

# Customize the ax1 plot
ax1.set(title="Heart Disease and Thalach Level",
       xlabel="Age",
       ylabel="Max Heart rate")

# Setting x axis limits (To remove extra spaces from tha x axis)
ax1.set_xlim([50,80]) # it is not a necessary here because ax0 and ax1 share same axis and we have done this same for ax0 above.


# Setting x axis limits (To remove extra spaces from tha y axis)
ax1.set_ylim([60,200])

# Add legend
ax1.legend(*scatter.legend_elements(), title="Target")

# Add Mean line
ax1.axhline(over_50["thalach"].mean(),linestyle="-.",color='r')

# Add a title to the figure
fig.suptitle("Heart Disease Analysis", fontsize=16, fontweight='bold');

