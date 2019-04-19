import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy

#Read the dataset
dataset = pd.read_csv("./dataset/yeast.data", header = None, sep = "\s+")
dataset.columns = ["Sequence Name", 
                    "mcg", 
                    "gvh", 
                    "alm", 
                    "mit", 
                    "erl", 
                    "pox", 
                    "vac", 
                    "nuc", 
                    "Class"
                    ]

#The column "Sequence Name" is just a database label that I'm not going to use to make predictions
dataset = dataset.drop(["Sequence Name"], axis = 1)

### NEURAL NETWORK ###

# Random seed for reproducibility
numpy.random.seed(7)

# Split the data in train and test datasets
train, test = train_test_split(dataset, test_size=0.2)

# Train
target_train = train.iloc[:, -1].values # Take only the last colum which is the value we want to predict
predictors_train = train.iloc[:, :8].values # Same as before but the opposite, remove last column

#Test
target_test = test.iloc[:, -1].values
predictors_test = test.iloc[:, :8].values

# In order for the NN to make a classification prediction, a labeled target needs to be a number,
# which is then converted to categorical.
le = LabelEncoder()

target_train = le.fit_transform(target_train)
target_train = to_categorical(target_train)

target_test = le.fit_transform(target_test)
target_test = to_categorical(target_test)

# Create the model
n_cols = 8 #number of predictors
model = Sequential()

model.add(Dense(100, activation = "relu", input_shape = (n_cols,)))
model.add(Dense(100, activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dense(10, activation = "softmax")) # softmax because it's a classification prediction

# Compile and fit the model
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
model.fit(predictors_train, target_train, epochs = 50, validation_data = (predictors_test, target_test))

# Make predictions
predictions = model.predict(predictors_test)