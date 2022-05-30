from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, BatchNormalization, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import numpy as np
import codecs
import json

warnings.filterwarnings("ignore")


def load_preprocess(data_path):
    data = pd.read_csv(data_path)
    data = data.to_numpy()
    np.random.shuffle(data)
    x = data[:, 1:].reshape(-1, 28, 28, 1) / 255.0
    y = data[:, 0].astype(np.int32)
    y = to_categorical(y, num_classes=len(set(y)))

    return x, y


train_path = "data/mnist_train.csv"
test_path = "data/mnist_test.csv"

x_train, y_train = load_preprocess(train_path)
x_test, y_test = load_preprocess(test_path)

index = 75
vis = x_train.reshape(60000, 28, 28)
plt.imshow(vis[index, :, :])
plt.legend()
plt.axis("off")
plt.show()
print(np.argmax(y_train[index]))

# className = glob(train_path + '/*')
numberofclass = y_train.shape[1]

# sıralı yapı olduğu için sequential
model = Sequential()
# conv layer ekle parametreler 32 filtre, 3x3 boyut, input shape belirle
model.add(Conv2D(input_shape=(28, 28, 1), filters=16, kernel_size=(3, 3)))
# nomalization için
model.add(BatchNormalization())
# reluyu ekle (aktivasyon fonk)
model.add(Activation("relu"))
# pooling layer ekle
model.add(MaxPooling2D())
# aynısından devam et
model.add(Conv2D(kernel_size=(3, 3), filters=32))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())
# layer arttırma (64)
model.add(Conv2D(filters=32, kernel_size=(3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())
# flatten ekle(düzleştirme)
model.add(Flatten())
# Dense layer ekle
model.add(Dense(units=256))  # output
# aktivasyon ekle
model.add(Activation("relu"))
# %50 al
model.add(Dropout(0.5))

model.add(Dense(units=numberofclass))  # output
# softmaxi aktive et
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# belirlediğimiz sayıda her iterasyondaki resim sayısı

batch_size = 3200

# steps per epoch her epochta train edilecek resim sayısı
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=batch_size)

# modeli kaydet
model.save_weights("mnist.h5")

print(hist.history.keys())
plt.plot(hist.history["loss"], label="Train Loss")
plt.plot(hist.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["acc"], label="Train acc")
plt.plot(hist.history["val_acc"], label="Validation acc")
plt.legend()
plt.show()

# %% save history
with open("deneme.json", "w") as f:
    json.dump(hist.history, f)

# %% load history

with codecs.open("cnn_fruit_hist.json", "r", encoding="utf-8") as f:
    h = json.loads(f.read())
plt.plot(h["loss"], label="Train Loss")
plt.plot(h["val_loss"], label="Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(h["acc"], label="Train acc")
plt.plot(h["val_acc"], label="Validation acc")
plt.legend()
plt.show()
