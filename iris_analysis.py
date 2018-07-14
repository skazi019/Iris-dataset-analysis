
# coding: utf-8

# In[ ]:


# Loading iris dataset from scikit-learn.


# In[1]:


from sklearn.datasets import load_iris


# In[2]:


# saving 'bunch' object containing iris dataset and its attributes.
iris = load_iris()
type(iris)


# In[3]:


# printing iris dataset. (each row represents a flower and each column represents the length and the width)
print(iris.data)
iris.data.shape


# In[4]:


# printing name of the four features.
print(iris.feature_names)


# In[5]:


# printing the integers representing the species of each observation.
print(iris.target)


# In[6]:


# printing the encoding scheme for species. (0 = setosa, 1 = versicolor, 2 = virginica)
print(iris.target_names)


# In[7]:


# checking the types of features and response.
type('iris.data')
type('iris.target')


# In[8]:


# checking the shape of the features. (first parameter is rows/observations, second parameter is columns/number of features)
iris.data.shape


# In[ ]:


# checking the shape of response. (one parameter matching the nuber of observations)
iris.target.shape


# # Scatter Plot with iris Dataset.

# In[11]:


# extracting the values of features and creating a list called featuresALL.
featuresALL = []
features = iris.data[:, [0,1,2,3]]
features.shape

