import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import confusion_matrix
from efficientnet.tfkeras import EfficientNetB0

_IMG_SIZE = 200
plt_ext = '.png'


def img_size():
    return _IMG_SIZE


def train_dir(_type):
    return os.path.join('test-' + _type, 'train')


def validation_dir(_type):
    return os.path.join('test-' + _type, 'validation')


def count_files(dirname):
    return sum([len(files) for r, d, files in os.walk(dirname)])


def get_base_model(modelname):
    if modelname == 'vgg16':
        print('MODELO CARGADO: VGG16')
        return tf.keras.applications.VGG16(input_shape=(img_size(), img_size(), 3),
                                           include_top=False,
                                           weights='imagenet')
    elif modelname == 'mobilenet':
        print('MODELO CARGADO: MOBILENET')
        return tf.keras.applications.MobileNetV2(input_shape=(img_size(), img_size(), 3),
                                                 include_top=False,
                                                 weights='imagenet')
    elif modelname == 'efficientnet':
        print('MODELO CARGADO: EFFICIENTNET')
        return EfficientNetB0(input_shape=(img_size(), img_size(), 3),
                              include_top=False,
                              weights='imagenet')


def get_model(_type, base_model, dropout):
    dense = None
    if _type == 'binary':
        dense = 1
    elif _type == 'categorical':
        dense = 4

    return tf.keras.Sequential([
        base_model,
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(dense, activation='softmax')
    ])


def show_model_layers(base_model):
    print()
    print("CAPAS EN EL MODELO SELECCIONADO: ", len(base_model.layers))
    print()


def get_loss(_type):
    if _type == 'binary':
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)
    elif _type == 'categorical':
        return tf.keras.losses.CategoricalCrossentropy()


def get_save_callback(modelname):
    return tf.python.keras.callbacks.ModelCheckpoint(modelname + '.h5',
                                                     monitor='val_accuracy',
                                                     save_best_only=True,
                                                     mode='max')


def get_generators(_type, batch_size):
    if _type != 'binary' and _type != 'categorical':
        print('TIPO NO VÁLIDO')
        return

    print('Imágenes de entrenamiento: ', count_files(train_dir(_type)))
    print('Imágenes de validación: ', count_files(validation_dir(_type)))

    image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    train_generator = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                          directory=train_dir(_type),
                                                          shuffle=True,
                                                          target_size=(img_size(), img_size()),
                                                          class_mode='categorical')

    image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    validation_generator = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                             directory=validation_dir(_type),
                                                             target_size=(img_size(), img_size()),
                                                             shuffle=False,
                                                             class_mode='categorical')
    return train_generator, validation_generator


def get_labels(_type):
    if _type == 'binary':
        return ['benigna', 'maligna']
    elif _type == 'categorical':
        return ['altogrado', 'ascus', 'bajogrado', 'benigna']


def get_confusion_matrix(_type, model):
    total_images = count_files(validation_dir(_type))

    image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    generator = image_gen_val.flow_from_directory(batch_size=total_images,
                                                  directory=validation_dir(_type),
                                                  target_size=(img_size(), img_size()),
                                                  shuffle=False,
                                                  class_mode=_type)
    return confusion_matrix(
        generator.classes,
        model.predict_classes(next(generator))
    )


def plot_history(modelname, history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'])
    axs[0].plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'])
    axs[0].set_title('Modelo de precisión')
    axs[0].set_ylabel('Precision')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(numpy.arange(1, len(history.history['accuracy']) + 1),
                      len(history.history['accuracy']) / 10)
    axs[0].legend(['train', 'val'], loc='best')

    axs[1].plot(range(1, len(history.history['loss']) + 1), history.history['loss'])
    axs[1].plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'])
    axs[1].set_title('Modelo de pérdidas')
    axs[1].set_ylabel('Pérdidas')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(numpy.arange(1, len(history.history['loss']) + 1), len(history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    filename = modelname + '-history'
    if os.path.isfile(filename + plt_ext):
        filename = filename + 'v2'
    filename = filename + plt_ext
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(modelname, _type, model):
    cm = get_confusion_matrix(_type, model)
    labels = get_labels(_type)

    plt.imshow(cm, interpolation='nearest')
    plt.title("Matriz de confusión")
    plt.colorbar()
    tick_marks = numpy.arange((len(labels)))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Reales')
    plt.xlabel('Predichos')
    filename = modelname + '-confusion-matrix'
    if os.path.isfile(filename + plt_ext):
        filename = filename + 'v2'
    filename = filename + plt_ext
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
