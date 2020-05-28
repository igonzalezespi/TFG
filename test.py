import utils

import tensorflow as tf

TYPE = 'categorical'

####### VARIABLES #######

MODELNAME = 'efficientnet'

epochs = 5
batch_size = 23
dropout = 0.5
learning_rate = 0.0006

fine_tune_epochs = 13
fine_tune_layers = 120
fine_tune_learning_rate = learning_rate / 10

#########################

use_fine_tune = True

print_model_summary = False

IMG_SIZE = utils.img_size()
TRAIN_DIR = utils.train_dir(TYPE)
VALIDATION_DIR = utils.validation_dir(TYPE)

loss = utils.get_loss(TYPE)
optimizer = tf.keras.optimizers.Adam()
fine_tune_optimizer = tf.keras.optimizers.Adam(lr=fine_tune_learning_rate)

total_epochs = epochs + fine_tune_epochs

# EMPIEZA

train_generator, validation_generator = utils.get_generators(TYPE, batch_size)

base_model = utils.get_base_model(MODELNAME)

base_model.trainable = False

if print_model_summary:
    base_model.summary()

model = utils.get_model(TYPE, base_model, dropout)

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

if print_model_summary:
    model.summary()

history = model.fit(train_generator,
                    epochs=epochs,
                    steps_per_epoch=train_generator.samples // train_generator.batch_size,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // validation_generator.batch_size)
if use_fine_tune:
    base_model.trainable = True

    utils.show_model_layers(base_model)

    for layer in base_model.layers[:fine_tune_layers]:
        layer.trainable = False

    model.compile(loss=loss,
                  optimizer=fine_tune_optimizer,
                  metrics=['accuracy'])

    if print_model_summary:
        model.summary()

    history_fine = model.fit(train_generator,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             steps_per_epoch=train_generator.samples // train_generator.batch_size,
                             validation_data=validation_generator,
                             validation_steps=validation_generator.samples // validation_generator.batch_size)
    utils.plot_history(MODELNAME, history_fine)
else:
    utils.plot_history(MODELNAME, history)

utils.plot_confusion_matrix(MODELNAME, TYPE, model)
