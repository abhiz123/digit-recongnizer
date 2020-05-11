import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
#%matplotlib inline

train = pd.read_csv("train.csv")

Y_train = train["label"]
X_train = train.drop("label",axis = 1)

del train

g = sns.countplot(Y_train)

# Y_train.value_counts()
plt.show()
