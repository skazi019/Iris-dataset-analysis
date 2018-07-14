
# coding: utf-8

# # Importing the iris Dataset.

# In[2]:


# Loading iris dataset from scikit-learn.


# In[3]:


from sklearn.datasets import load_iris


# In[4]:


# saving 'bunch' object containing iris dataset and its attributes.
iris = load_iris()
type(iris)


# In[8]:


# printing iris dataset. (each row represents a flower and each column represents the length and the width)
print(iris.data)
iris.data.shape


# In[10]:


# printing name of the four features.
print(iris.feature_names)


# In[11]:


# printing the integers representing the species of each observation.
print(iris.target)


# In[12]:


# printing the encoding scheme for species. (0 = setosa, 1 = versicolor, 2 = virginica)
print(iris.target_names)


# In[13]:


# checking the types of features and response.
type('iris.data')
type('iris.target')


# In[14]:


# checking the shape of the features. (first parameter is rows/observations, second parameter is columns/number of features)
iris.data.shape


# In[15]:


# checking the shape of response. (one parameter matching the nuber of observations)
iris.target.shape


# # Scatter Plot with iris Dataset.

# In[16]:


# extracting the values of features and creating a list called featuresALL.
featuresALL = []
features = iris.data[:, [0,1,2,3]]
features.shape


# In[17]:


# extracting the values for target.
targets = iris.target
targets.reshape(targets.shape[0],-1)
targets.shape


# In[19]:


# every observation gets appended into the list once it's read. 'For' loop is used for iteration process.
for observation in features:
    featuresALL.append(observation[0] + observation[1] + 
                       observation[2] + observation[3])
print(featuresALL)


# In[22]:


# plotting the Scatter Plot
import matplotlib.pyplot as plt
plt.scatter(featuresALL, targets, color='red', alpha=1.0)
plt.rcParams['figure.figsize'] = [10,8] #re-sizing the figure plotted.
plt.title('Iris dataset scatter plot')
plt.xlabel('Features')
plt.ylabel('Targets')
plt.show()


# ### Scatter Plot with iris Dataset. (Relationship between Sepal Length and Sepal Width) Method #1
# 

# In[26]:


# finding relationship between Sepal length and Sepal width.
featuresALL = []
targets = []
for feature in features:
    featuresALL.append(feature[0]) #Sepal length
    targets.append(feature[1]) #Sepal width
groups = ('Iris-setosa', 'Tris-versicolor', 'Iris-virginica')
colors = ('blue', 'green', 'red')
data = ((featuresALL[:50], targets[:50]), (featuresALL[50:100], targets[50:100]), 
         (featuresALL[100:150], targets[100:150]))
for item, colors, groups in zip(data, colors,groups):
    #item = (featuresALL[:50], targets[:50]), (featuresALL[50:100], targets[50:100]), 
    #     (featuresALL[100:150], targets[100:150])
    x, y = item
    plt.scatter(x, y, color=colors, alpha=1)
    plt.title('Iris Dataset scatter Plot')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()


# ### Scatter Plot with Iris Dataset (Relationship between Petal length and Petal width) #Method 1

# In[27]:


# finding relationship between Sepal length and Sepal width.
featuresALL = []
targets = []
for feature in features:
    featuresALL.append(feature[2]) #Pepal length
    targets.append(feature[3]) #Pepal width
groups = ('Iris-setosa', 'Tris-versicolor', 'Iris-virginica')
colors = ('blue', 'green', 'red')
data = ((featuresALL[:50], targets[:50]), (featuresALL[50:100], targets[50:100]), 
         (featuresALL[100:150], targets[100:150]))
for item, colors, groups in zip(data, colors,groups):
    #item = (featuresALL[:50], targets[:50]), (featuresALL[50:100], targets[50:100]), 
    #     (featuresALL[100:150], targets[100:150])
    x0, y0 = item
    plt.scatter(x0, y0, color=colors, alpha=1)
    plt.title('Iris Dataset scatter Plot')
plt.xlabel('pepal length')
plt.ylabel('pepal width')
plt.show()


# # K - Nearest neighbours (KNN) Algorithm

# In[30]:


import pandas as pd
iris = load_iris()
ir = pd.DataFrame(iris.data)
ir.columns = iris.feature_names
ir['CLASS'] = iris.target
ir.head() #returns the top 5 rows.


# In[32]:


from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(5) #The arguement specify to return the fast 5 among the
#dataset

nn.fit(iris.data) #fitting iris dataset to nearest neighbors algorithm


# In[33]:


ir.describe() #showing the fitted data


# In[34]:


# creating a test data
import numpy as np
test = np.array([5.4,2,2,2.3])
test1 = test.reshape(1,-1)
test1.shape


# In[35]:


nn.kneighbors(test1,5)


# In[36]:


ir.iloc[[98, 93, 57, 60, 79]] # displaying specific rows using iloc() 


# ### KNeighborsClassifier Algorithm
