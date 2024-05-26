from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess
from keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess
from keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset settings
im_shape = (224, 224)
TRAINING_DIR = 'images'
TEST_DIR = 'images'
seed = 10
BATCH_SIZE = 16

# Data Augmentation
data_generator = ImageDataGenerator(
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    preprocessing_function=inception_preprocess,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_data_generator = ImageDataGenerator(preprocessing_function=inception_preprocess, validation_split=0.2)

train_generator = data_generator.flow_from_directory(
    TRAINING_DIR, target_size=im_shape, shuffle=True, seed=seed,
    class_mode='categorical', batch_size=BATCH_SIZE, subset="training"
)
validation_generator = val_data_generator.flow_from_directory(
    TRAINING_DIR, target_size=im_shape, shuffle=False, seed=seed,
    class_mode='categorical', batch_size=BATCH_SIZE, subset="validation"
)
test_generator = ImageDataGenerator(preprocessing_function=inception_preprocess).flow_from_directory(
    TEST_DIR, target_size=im_shape, shuffle=False, seed=seed,
    class_mode='categorical', batch_size=BATCH_SIZE
)

nb_train_samples = train_generator.samples
nb_validation_samples = validation_generator.samples
nb_test_samples = test_generator.samples
classes = list(train_generator.class_indices.keys())
num_classes = len(classes)

# Data Visualization
plt.figure(figsize=(15, 15))
for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = (train_generator.next()[0] + 1) / 2 * 255
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()

# Model Training Function
def train_model(base_model, preprocess_input, model_name):
    base_model = base_model(weights='imagenet', include_top=False, input_shape=(im_shape[0], im_shape[1], 3))

    x = base_model.output
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax', kernel_initializer='random_uniform')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    base_model.trainable = False

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks_list = [
        keras.callbacks.ModelCheckpoint(filepath=f'models/{model_name}.h5', monitor='val_loss', save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    ]

    history = model.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // BATCH_SIZE,
        epochs=20,
        callbacks=callbacks_list,
        validation_data=validation_generator,
        verbose=1,
        validation_steps=nb_validation_samples // BATCH_SIZE
    )

    # Plotting Training History
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs_x = range(1, len(loss_values) + 1)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_x, loss_values, 'bo', label='Training loss')
    plt.plot(epochs_x, val_loss_values, 'b', label='Validation loss')
    plt.title(f'{model_name} Training and validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    plt.subplot(2, 1, 2)
    plt.plot(epochs_x, acc_values, 'bo', label='Training acc')
    plt.plot(epochs_x, val_acc_values, 'b', label='Validation acc')
    plt.title(f'{model_name} Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()

    model = load_model(f'models/{model_name}.h5')
    val_score = model.evaluate(validation_generator)
    test_score = model.evaluate(test_generator)
    return val_score, test_score

# Training models and evaluating their performance
inception_val_score, inception_test_score = train_model(InceptionV3, inception_preprocess, 'InceptionV3')
vgg19_val_score, vgg19_test_score = train_model(VGG19, vgg19_preprocess, 'VGG19')
resnet50_val_score, resnet50_test_score = train_model(ResNet50, resnet_preprocess, 'ResNet50')

# Analysis
analysis = pd.DataFrame({
    'Model': ['InceptionV3', 'VGG19', 'ResNet50'],
    'Val_Loss': [inception_val_score[0], vgg19_val_score[0], resnet50_val_score[0]],
    'Val_Accuracy': [inception_val_score[1], vgg19_val_score[1], resnet50_val_score[1]],
    'Test_Loss': [inception_test_score[0], vgg19_test_score[0], resnet50_test_score[0]],
    'Test_Accuracy': [inception_test_score[1], vgg19_test_score[1], resnet50_test_score[1]]
})
print(analysis)
