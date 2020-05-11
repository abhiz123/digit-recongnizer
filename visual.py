import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
#%matplotlib inline

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

Y_train = train["label"]
X_train = train.drop("label",axis = 1)
del train

g = sns.countplot(Y_train)

#normalizing data so CNN's converge faster
X_train = X_train / 255.0
test = test / 255.0

#reshaping because pandas require 28*28
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
