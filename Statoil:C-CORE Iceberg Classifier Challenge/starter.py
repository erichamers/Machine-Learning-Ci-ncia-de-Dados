## ---- Imports ---- ## 
import numpy as np 
import pandas as pd

from keras.models import Sequential
from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense, Dropout

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

## ---- Data Import ---- ##
train_df = pd.read_json('../input/train.json')
test_df = pd.read_json('../input/test.json')

## ---- Data Preparation ---- ##
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]])

X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
y_train = np.array(train_df['is_iceberg'])
print('X_train Shape:', X_train.shape)

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_2"]])
X_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
print('X_test Shape:', X_test.shape)

## ---- Network Architecture ---- ##
model = Sequential()
model.add(Convolution2D(32, 3, activation='relu', input_shape=(75, 75, 2)))
model.add(Convolution2D(64, 3, activation='relu', input_shape=(75, 75, 2)))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.summary()

## ---- Fit the model ---- ## 
model.fit(X_train, y_train, validation_split=0.2)

# Prediction for kaggle submission
prediction = model.predict(X_test, verbose=1)

# Kaggle submission file
submit_df = pd.DataFrame({'id': test_df["id"], 'is_iceberg': prediction.flatten()})
submit_df.to_csv('./kerasmodel.csv', index=False)