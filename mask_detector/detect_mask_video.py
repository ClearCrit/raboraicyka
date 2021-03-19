from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import logging
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, filename='shape_log_{:%Y_%m_%d}.log'.format(datetime.now()),
                    format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


def shape_log(detections):
    logger.info("shape {}".format(detections.shape))
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def detect_and_predict_mask(frame, face_net, mask_net):
    # создание блоба по размеру фрейма
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # прогонка блоба через нейронку для определения лица
    face_net.setInput(blob)
    detections = face_net.forward()
    shape_log(detections)

    # инициализация списка лиц и их месторасположения, + список прогнозов из нашей нейронки
    faces = []
    coordinates = []
    probabilities = []

    # постоянно крутить обнаружение
    for i in range(0, detections.shape[2]):
        # сохранение вероятности обнаружения
        confidence = detections[0, 0, i, 2]

        # фильтрация кривого обнаружения, для повышения точности confidence
        if confidence > 0.5:
            # вычисляем (x, y)координаты при ограничениях прямоугольника для объекта
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")

            # проверка, что ограничивающие рамки не выходят за пределы фрейма
            startX, startY = (max(0, startX), max(0, startY))
            endX, endY = (min(w - 1, endX), min(h - 1, endY))

            # извлекаем ROI(область интереса), конвертируем из BGR to RGB, меняем размер на 224х224 и в обработку
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # падает
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # добавление граней и рамок ограничений к соотвестющему листу
            faces.append(face)
            coordinates.append((startX, startY, endX, endY))

    # делать прогноз, если хоть 1 лицо обнаружено
    if len(faces) > 0:
        # чтобы быстрее работало, удобнее делать прогнозы для всех лиц в одно время, чем по очередно
        faces = np.array(faces, dtype="float32")
        probabilities = mask_net.predict(faces, batch_size=32)

    # возвращение кортежа местоположения лиц и их текущего расположения
    return coordinates, probabilities


if __name__ == "__main__":
    # подгрузка сериализованной обученной модели и её десериализация
    proto_txt_path = r"face_detector\deploy.prototxt"
    weights_path = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
    face_net = cv2.dnn.readNet(proto_txt_path, weights_path)

    # подгрузка модели
    mask_net = load_model("mask_detector.model")

    # инициализация потока видео
    logger.info("starting video stream...")
    vs = VideoStream(src=0).start()

    # циклю при рабочем потоке видео
    while True:
        # захват видео с потока и ресайз до максимальных 400 пикселей
        frame = imutils.resize(vs.read(), width=400)

        # определение лиц на видеопотоке и определение есть ли на них маска
        try:
            locs, preds = detect_and_predict_mask(frame, face_net, mask_net)  # падает
            for box, pred in zip(locs, preds):
                # распаковка ограничивающей рамке и прогнозов
                startX, startY, endX, endY = box
                mask, withoutMask = pred

                # определение лейбла и цвета для отрисовки ограничевающих рамок
                label, color = ("In mask", (0, 255, 0)) if mask > withoutMask else ("Without mask", (0, 0, 255))

                # добавление вероятности в лейбл
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # отображение лейбла и ограничивающих рамок поверх видеопотока
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # вывод
            cv2.imshow("Video", frame)
            key = cv2.waitKey(1) & 0xFF

            # остановить видео поток, если нажать q
            if key in (ord("q"), ord("Q")):
                break
        except Exception as e:
            logger.error(e)
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        # перебор всех лиц и их текущего местарасположения на видео

    # отчистка
    cv2.destroyAllWindows()
    vs.stop()
