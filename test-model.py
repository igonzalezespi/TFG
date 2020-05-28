import utils
import os

import tensorflow as tf

####### VARIABLES #######

TYPE = 'categorical'
MODELNAME = 'efficientnet'

#########################

IMG_SIZE = utils.img_size()
TRAIN_DIR = utils.train_dir(TYPE)
VALIDATION_DIR = utils.validation_dir(TYPE)

model = tf.keras.models.load_model(os.path.join('backup-categorical', 'efficientnetb0.h5'))

utils.plot_confusion_matrix(MODELNAME, TYPE, model)
