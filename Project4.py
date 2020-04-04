from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

FEATURES = 57
np.random.seed(1)
file = 'spam.txt'
data = np.loadtxt(file, delimiter=' ', dtype=float)

X = data[:, :-1]
y = data[:, -1]

Xm = np.zeros(len(y))
Xs = np.zeros(len(y))
for i in range(np.shape(X)[1]):
    Xm = np.mean(X[:, i])
    Xs = np.std(X[:, i], ddof=1)
    X[:, i] = (X[:, i]-Xm)/Xs
'''
for j in range(len(y)):
    if y[j] == 0:
        y[j] = -1
'''
indices = np.random.permutation(X.shape[0])
num = X.shape[0] // 5

rg = indices[0:num]
X_test = X[rg][:]
y_test = y[rg]

X_train = np.delete(X, rg, axis=0)
y_train = np.delete(y, rg, axis=0)

model1 = keras.Sequential([
    keras.layers.Dense(10, activation='sigmoid', use_bias='FALSE', input_shape=(FEATURES,)),
    keras.layers.Dense(1, activation = 'sigmoid', use_bias='FALSE')
])
model1.summary()
model1.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history1 = model1.fit(X_train, y_train, epochs = 100, validation_split = 0.3, verbose=1)

model2 = keras.Sequential([
    keras.layers.Dense(100, activation='sigmoid', use_bias='FALSE', input_shape=(FEATURES,)),
    keras.layers.Dense(1, activation = 'sigmoid', use_bias='FALSE')
])
model2.summary()
model2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history2 = model2.fit(X_train, y_train, epochs = 100, validation_split = 0.3, verbose=1)

model3 = keras.Sequential([
    keras.layers.Dense(1000, activation='sigmoid', use_bias='FALSE', input_shape=(FEATURES,)),
    keras.layers.Dense(1, activation = 'sigmoid', use_bias='FALSE')
])
model3.summary()
model3.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history3 = model3.fit(X_train, y_train, epochs = 100, validation_split = 0.3, verbose=1)

history_dict = history1.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
b1 = val_loss.index(min(val_loss))+1
epochs = range(1, len(acc) + 1)

# “bo”代表 "蓝点"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b代表“蓝色实线”
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

history_dict = history2.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
b2 = val_loss.index(min(val_loss))+1
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'ro', label='Training loss')
# b代表“蓝色实线”
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

history_dict = history3.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
b3 = val_loss.index(min(val_loss))+1
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'go', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model1.fit(X_train, y_train, epochs = b1, validation_split = 0.3, verbose=1)
model2.fit(X_train, y_train, epochs = b2, validation_split = 0.3, verbose=1)
model3.fit(X_train, y_train, epochs = b3, validation_split = 0.3, verbose=1)
test_loss, test_acc = model1.evaluate(X_test,  y_test, verbose=0)
print('\nTest epoch:', b1)
print('\nTest accuracy:', test_acc)
test_loss, test_acc = model2.evaluate(X_test,  y_test, verbose=0)
print('\nTest epoch:', b2)
print('\nTest accuracy:', test_acc)
test_loss, test_acc = model3.evaluate(X_test,  y_test, verbose=0)
print('\nTest epoch:', b3)
print('\nTest accuracy:', test_acc)
'''plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
'''