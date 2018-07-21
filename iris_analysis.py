
# coding: utf-8

# # Importing the iris Dataset.

# In[3]:


# Loading iris dataset from scikit-learn.


# In[4]:


from sklearn.datasets import load_iris


# In[5]:


# saving 'bunch' object containing iris dataset and its attributes.
iris = load_iris()
type(iris)


# In[6]:


# printing iris dataset. (each row represents a flower and each column represents the length and the width)
print(iris.data)
iris.data.shape


# In[7]:


# printing name of the four features.
print(iris.feature_names)


# In[8]:


# printing the integers representing the species of each observation.
print(iris.target)


# In[9]:


# printing the encoding scheme for species. (0 = setosa, 1 = versicolor, 2 = virginica)
print(iris.target_names)


# In[10]:


# checking the types of features and response.
type('iris.data')
type('iris.target')


# In[11]:


# checking the shape of the features. (first parameter is rows/observations, second parameter is columns/number of features)
iris.data.shape


# In[12]:


# checking the shape of response. (one parameter matching the nuber of observations)
iris.target.shape


# # Scatter Plot with iris Dataset.

# In[13]:


# extracting the values of features and creating a list called featuresALL.
featuresALL=[]
features = iris.data[:, [0,1,2,3]]
features.shape


# In[14]:


# extracting the values for target.
targets = iris.target
targets.reshape(targets.shape[0],-1)
targets.shape


# In[15]:


# every observation gets appended into the list once it's read. 'For' loop is used for iteration process.
for observation in features:
    featuresALL.append(observation[0] + observation[1] + 
                       observation[2] + observation[3])
print(featuresALL)


# In[16]:


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

# In[17]:


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

# In[18]:


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

# In[19]:


import pandas as pd
iris = load_iris()
ir = pd.DataFrame(iris.data)
ir.columns = iris.feature_names
ir['CLASS'] = iris.target
ir.head() #returns the top 5 rows.


# In[20]:


from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(5) #The arguement specify to return the fast 5 among the
#dataset

nn.fit(iris.data) #fitting iris dataset to nearest neighbors algorithm


# In[21]:


ir.describe() #showing the fitted data


# In[22]:


# creating a test data
import numpy as np
test = np.array([5.4,2,2,2.3])
test1 = test.reshape(1,-1)
test1.shape


# In[23]:


nn.kneighbors(test1,5)


# In[24]:


ir.iloc[[98, 93, 57, 60, 79]] # displaying specific rows using iloc() 


# ### KNeighborsClassifier Algorithm

# In[25]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15

# we only take the first two features. We could avoid this ugly slicing by using 
# a two-dim dataset
x = iris.data[:, :2]
y = iris.target

h = 0.02 #step size in the mesh

#creating color map. cmap maps numbers(which are normalized between [0,1]) to colors.
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']) 
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # creating an instance of Neighbors Classifier and fitting the data
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(x, y)
    
    # for plotting the decision boundary we will assign a color to each point
    # in the mesh [x_min, x_max]x[y_min, y_max]
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), 
                        np.arange(y_min, y_max, h))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # putting the result into a colorplot
    z = z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, z, cmap=cmap_light)
    
    # also plotting training points
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, wights = '%s')" 
              % (n_neighbors, weights))
    
plt.show()


# ### KNN Classifier Algorithm - Understanding its working

# In[26]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
print(knn)


# In[27]:


import numpy as np
X1 = np.asarray(featuresALL)
X1 = X1.reshape(-1,1)


# In[28]:


X1.shape


# In[29]:


y = iris.target
y.shape


# In[30]:


knn.fit(X1,y)


# In[31]:


import numpy as np
print(knn.predict([[6.4]]))


# In[32]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X1,y)


# In[33]:


print(knn.predict([[3.4]]))


# In[34]:


print(knn.predict(np.column_stack([[1.,6.1,3.2,4.2]])))


# # Linear Regression

# In[35]:


from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model


# In[36]:


import numpy as np
XX = np.asarray(featuresALL)
X2 = XX[:, np.newaxis]
X2
X2.shape


# In[37]:


y2 = iris.target
y2.shape


# In[38]:


model.fit(X2, y2)


# In[39]:


model.coef_


# In[40]:


model.intercept_


# In[41]:


Xfit = np.random.randint(8,size=(150))
Xfit.astype(float)
Xfit = Xfit[:, np.newaxis]
Xfit.shape


# In[42]:


yfit = (model.predict(Xfit))
yfit.shape


# In[43]:


plt.scatter(X2, y2)
plt.plot(Xfit, yfit)


# ### Regression

# In[44]:


from sklearn.preprocessing import PolynomialFeatures
poly =PolynomialFeatures(150, include_bias=False)
poly.fit_transform(X2)


# In[45]:


from sklearn.pipeline import make_pipeline
poly_model = make_pipeline(PolynomialFeatures(3), LinearRegression())
poly_model.fit(X2, y2)
yfit = poly_model.predict(Xfit)


# In[46]:


# this linear model, through the use of 3rd order polynomial basis function
# can provide a fit to this non-linear data
plt.scatter(X2, y2)
plt.plot(Xfit, yfit)


# ### How length and width vary according to the species

# In[47]:


import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
iris1 = pd.read_csv(url, names=names)
print(iris1.head())


# ### Scatter Plot with Iris Dataset (Relationship between Sepal length and Sepal Width)

# In[48]:


iris1.plot(kind ='scatter', x='SepalLengthCm', y='SepalWidthCm')
plt.show()


# ### Scatter Plot with Iris Dataset (Relationship between Petal length and Petal width)

# In[49]:


iris1.plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm')
plt.show()


# In[50]:


iris1.ix[:, iris1.columns].hist()
plt.figure(figsize=(15,10))
plt.show()


# ### Violin Plot

# In[51]:


import seaborn as sns
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species', y='PetalLengthCm',data=iris1)
plt.subplot(2,2,2)
sns.violinplot(x='Species', y='PetalWidthCm',data=iris1)
plt.subplot(2,2,3)
sns.violinplot(x='Species', y='SepalLengthCm',data=iris1)
plt.subplot(2,2,4)
sns.violinplot(x='Species', y='SepalWidthCm',data=iris1)


# ### IRIS Correlation Matrix

# In[52]:


corr = iris1.corr()
corr


# In[55]:


# importing correlation matrix to see parameters which best correlate each other.
# Accroding to the correlation matrix results PetalLengthCm and PetalWidthCm
# have a positive correlation which is proved by the scatter plot discussed above

import seaborn as sns
import pandas as pd
corr = iris1.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values,
           cmap='viridis', annot=True)
plt.show()

