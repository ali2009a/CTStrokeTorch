from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
from my_metrics import dice_coefficient, dice_coefficient_loss

#K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')

def load_old_model(model_file):
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss,'dice_coefficient': dice_coefficient}
    return load_model(model_file, custom_objects)

def train_model(model, model_file, training_generator, validation_generator, steps_per_epoch, validation_steps,
                initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                learning_rate_patience=20, early_stopping_patience=None):
    model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True)
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        callbacks= [model_checkpoint])

