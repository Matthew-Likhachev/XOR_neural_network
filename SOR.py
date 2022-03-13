import keras as k
import pandas as pd
import numpy  as np
from keras import models
from keras import layers
import matplotlib .pyplot as plt

x_train=np.array([[0,0], [0,1], [1,0], [1,1]])
y_train=np.array([[0], [0], [1], [0]])

#print(x_train[0])
x_test=np.array([[0,0], [0,1], [1,0], [1,1]])
y_test=np.array([[0], [0], [1], [0]])



model = models.Sequential()
model.add(layers.Dense(8,activation='tanh'))
#model.add(layers.Dense(8, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

history = model.fit(x_train,
  y_train,
  epochs=200,
  batch_size=1,
  validation_data=(x_test, y_test))

history_dict = history.history
#print(history_dict.keys())
#>>dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['accuracy'])+ 1)

#plt.plot( )

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss') 
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.clf() #Очистить рисунок
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, history_dict['accuracy'], 'bo', label='Training acc')
plt.plot(epochs, history_dict['val_accuracy'], 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print(model.predict(x_test))
