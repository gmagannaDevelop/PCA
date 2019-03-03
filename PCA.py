#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import random as rd

from sklearn.decomposition import PCA
from sklearn import preprocessing

import matplotlib.pyplot as plt
#import seaborn as sb


# In[36]:


# Create a sample dataset:
genes = ['gene' + str(i) for i in range(1, 101)]


# In[37]:


# Knockout and wildtype samples:
wt = ['wt' + str(i) for i in range(1, 6)]
ko = ['ko' + str(i) for i in range(1, 6)]


# In[38]:


# Added the * to unpack the values of both lists.
data = pd.DataFrame(columns=[*wt, *ko], index=genes)


# In[39]:


for gene in data.index:
    data.loc[gene, 'wt1':'wt5'] = np.random.poisson(lam=rd.randrange(10, 1000), size=5)
    data.loc[gene, 'ko1':'ko5'] = np.random.poisson(lam=rd.randrange(10, 1000), size=5)


# Note that this dataframe is the transpose of what we ususally work with on pandas. Here each row represents a dimension/variable and each column represents an observation.

# In[40]:


data.head()


# In[41]:


# Center and scale data to have average value of 0
# and a standard deviation of 1.

scaled_data = preprocessing.scale(data.T) 
# ^ a transpose is necessary due to our DataFrame.


# ## Note on scaling:
# Sklearn calculates the transformation using:
# $ \tilde{x} = \frac{(x - \bar{x})^2}{ n }$
# 
# While R uses:
# $ \tilde{x} = \frac{(x - \bar{x})^2 }{ n - 1 }$
# 
# R's method results in larger, but unbiased estimates of the variation.

# In[42]:


# Alternatively:
# scaled_data = StandardScaler().fit_transform(data.T)


# In[43]:


pca = PCA()


# In[44]:


# Calculate the loading scores and variation for each PC.
pca.fit(scaled_data)


# In[45]:


pca_data = pca.transform(scaled_data)


# In[46]:


# Calculate percent of variation for each Principal Component
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)


# In[47]:


# Create labels for the scree plot.
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)] 


# In[48]:


plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')


# As shown by the previous scree plot, almost all of the variation is along the first PC, so a 2-D graph, using PC1 and PC2 should do a good job representing the original data.

# In[49]:


pca_df = pd.DataFrame(pca_data, index=[*wt, *ko], columns=labels)


# In[50]:


plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA Graph')
plt.xlabel(f'PC1 - {per_var[0]}')
plt.ylabel(f'PC2 - {per_var[1]}')

for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))


# In[51]:


loading_scores = pd.Series(pca.components_[0], index=genes)
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
top_10_genes = sorted_loading_scores[0:10].index.values


# In[52]:


# Get the 'contribution score for each gene.'
loading_scores[top_10_genes]


# The values are super similar, so a lot of the genes played a role un separating the samples, rather than just one or two.

# In[ ]:




