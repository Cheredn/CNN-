import os
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Путь к датасету
DATASET_PATH = 'dataset/'

# Определим количество классов (папок)
num_classes = len(os.listdir(DATASET_PATH))

# Используем categorical class_mode для 3+ классов
class_mode = "categorical"

# Аугментация для тренировки и только rescale для валидации
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Датасеты
train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=16,
    class_mode=class_mode,
    subset='training'
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=16,
    class_mode=class_mode,
    subset='validation'
)

# Загрузка предобученной MobileNetV2 без верхнего слоя
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Замораживаем веса

# Добавим классификатор
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Обучение
model.fit(train_data,
          validation_data=val_data,
          epochs=30,
          callbacks=[early_stop])

# Оценка
test_loss, test_accuracy = model.evaluate(val_data)
print(f'Точность модели на валидационных данных: {test_accuracy:.2f}')

# Сохраняем модель
model.save('image_classifierV2.h5')
