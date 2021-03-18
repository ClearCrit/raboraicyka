from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# инициализируем начальную скорость обучения, количество эпох для обучения и размер пакета
EPOCHS = 20
BS = 32

direct = r"C:\Users\ClearCrit\Downloads\mask_detector\dataset"
categor= ["with_mask", "without_mask"]

# возьмем список изображений из набора данных, потому инициализируем список данных (т.е. изображений) и изображений классов
print("[INFO] loading images...")

data = []
labels = []

for category in categor:
    path = os.path.join(direct, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

# кодировка лейблов
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# создания тренировачного генератора для ++ скорости обработки
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# подгрузка MobileNetV2, не включать FC layer sets
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# создание хэда модели, которая будет поверх базовой модели
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# помещение хэда FC модели поверх базовой модели (теперь эту модель тренируем)
model = Model(inputs=baseModel.input, outputs=headModel)

# перебор всех слоев базовой модели и остановка(заморозка их), чтоб они не поменялись при первом обучении
for layer in baseModel.layers:
	layer.trainable = False

# компиляция модели
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# обучения хэда модели
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# создание прогноза на тестовый сет
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# для каждого изображения в тестовой выборке  нужно найти индекс метки с соответствующей наибольшей предсказанной вероятностью
predIdxs = np.argmax(predIdxs, axis=1)

#просто красивый отчёт о классификации 
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# серализация модели на диск 
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# типо красивый график потерь и точность, при обучении
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")