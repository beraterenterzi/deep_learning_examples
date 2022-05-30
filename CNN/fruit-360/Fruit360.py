
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, image_utils
from keras.utils import img_to_array, load_img
import matplotlib.pyplot as plt
from glob import glob

train_path = "Training/"
test_path = "Test/"

img = image_utils.load_img(path=train_path + "Apple Braeburn/0_100.jpg")

plt.imshow(img)
plt.axis("off")
plt.show()

x = img_to_array(img)
print(x.shape)

className = glob(train_path + '/*')
numberofclass = len(className)

# sıralı yapı olduğu için sequential
model = Sequential()
# conv layer ekle parametreler 32 filtre, 3x3 boyut, input shape belirle
model.add(Conv2D(32, (3, 3), input_shape=x.shape))
# reluyu ekle (aktivasyon fonk)
model.add(Activation("relu"))
# pooling layer ekle
model.add(MaxPooling2D())
# aynısından devam et
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())
# layer arttırma (64)
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())
# flatten ekle(düzleştirme)
model.add(Flatten())
# Dense layer ekle
model.add(Dense(1024))
# aktivasyon ekle
model.add(Activation("relu"))
# %50 al
model.add(Dropout(0.5))

model.add(Dense(numberofclass))  # output
# softmaxi aktive et
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

# belirlediğimiz sayıda her iterasyondaki resim sayısı
batch_size = 32

# Augmentation, Data generation (rescale tekrar boyutlandır, shear= çevirme oranı, horizantal= çevir, zoom= zoom yap)
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.3, horizontal_flip=True, zoom_range=0.6)
test_datagen = ImageDataGenerator(rescale=1. / 255)
# keras kendi generate ediyor
train_generator = train_datagen.flow_from_directory(train_path, target_size=x.shape[:2], batch_size=batch_size,
                                                    color_mode="rgb", class_mode="categorical")

test_generator = test_datagen.flow_from_directory(test_path, target_size=x.shape[:2], batch_size=batch_size,
                                                  color_mode="rgb", class_mode="categorical")

# steps per epoch her epochta train edilecek resim sayısı
hist = model.fit_generator(generator=train_generator, steps_per_epoch=1600 // batch_size, epochs=150,
                    validation_data=test_generator, validation_steps=900)

# modeli kaydet
model.save_weights("deneme.h5")

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
import json

with open("deneme.json", "w") as f:
    json.dump(hist.history, f)

# %% load history
import codecs

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
