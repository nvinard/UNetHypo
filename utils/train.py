import tensorflow as tf
import os
import datetime
import pathlib
from utils import model
from tensorflow.keras.callbacks import TensorBoard
import glob
import matplotlib.pyplot as plt
from utils import data_pipeline as dp


# Define metrics here
from tensorflow.keras import backend as K

def iou_coef_gaussian(y_true, y_pred, smooth=1):
    y_true = tf.dtypes.cast(y_true>0.1, tf.int32)
    y_true = tf.dtypes.cast(y_true, tf.float32)
    y_pred = tf.dtypes.cast(y_pred>0.1, tf.int32)
    y_pred = tf.dtypes.cast(y_pred, tf.float32)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice_coef_gaussian(y_true, y_pred, smooth=1):
    y_true = tf.dtypes.cast(y_true>0.1, tf.int32)
    y_true = tf.dtypes.cast(y_true, tf.float32)
    y_pred = tf.dtypes.cast(y_pred>0.1, tf.int32)
    y_pred = tf.dtypes.cast(y_pred, tf.float32)
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def train_and_evaluate(output_dir, cfg):

    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)

    # Define the checkpoint directory to store the checkpoints and create the directory if it does not already exist
    if os.path.exists(os.path.join(output_dir, cfg.MAIN_CHECKPOINTS)) == False:
        os.mkdir(os.path.join(output_dir, cfg.MAIN_CHECKPOINTS))

    if cfg.FROM_CHECKPOINT == False:
        # Create new checkpoint folder
        checkpoint_dir = os.path.join(output_dir, "training_checkpoints/training_{}_{}".format(cfg.LABEL,datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
        if os.path.exists(checkpoint_dir) == False:
            os.mkdir(checkpoint_dir)
    else:
        checkpoint_dir = os.path.join(output_dir, "training_checkpoints/training_{}".format(cfg.CHECK_FOLDER))

    log_dir = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    callbacks = [
                 #tf.keras.callbacks.TensorBoard(log_dir=output_dir +'/logs/'+log_dir,update_freq=5760),
                 tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
                 ]


    if cfg.FROM_CHECKPOINT:
        print("")
        print("Resuming training from latest checkpoint")
        checkpoint_dir = os.path.dirname(checkpoint_prefix)
        checkpoints = pathlib.Path(checkpoint_dir).glob("*.index")
        checkpoints = sorted(checkpoints, key=lambda cp:cp.stat().st_mtime)
        checkpoints = [cp.with_suffix('') for cp in checkpoints]
        latest = str(checkpoints[-1])
        print("LATEST checkpoint", latest)

        generator = model.GeneratorNew()
        generator.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy', iou_coef_gaussian, dice_coef_gaussian])
        generator.load_weights(latest)
    else:
        print("Starting training from scratch")
        generator = model.GeneratorNew()
        generator.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy', iou_coef_gaussian, dice_coef_gaussian])

    file_path = os.path.join(cfg.PATH_TO_DATA,'%s*' % cfg.TRAIN)
    file_list = tf.io.matching_files(file_path)
    n_train_examples = len(file_list)*400
    file_path_test = os.path.join(cfg.PATH_TO_DATA,'%s*' % cfg.TEST)
    file_list_test = tf.io.matching_files(file_path_test)
    n_test_examples = len(file_list_test)*400
    steps_per_epoch = n_train_examples // cfg.BATCH_SIZE
    test_steps = n_test_examples // cfg.BATCH_SIZE

    # Load the training data
    trainds = dp.read_dataset(prefix=cfg.TRAIN, batch_size=cfg.BATCH_SIZE, cfg=cfg)
    testds = dp.read_dataset(prefix=cfg.TEST, batch_size=cfg.BATCH_SIZE, cfg=cfg)

    history = generator.fit(
        trainds,
        validation_data=testds,
        epochs=cfg.EVAL_STEPS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=test_steps,
        callbacks=callbacks,
        verbose=2)

    save_as = os.path.join(output_dir, "{}_{}.h5".format(cfg.LABEL, datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
    generator.save(save_as, overwrite=True, include_optimizer=True)

    # Save accuracy and loss curves
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    iou_g = history.history['iou_coef_gaussian']
    dice_g = history.history['dice_coef_gaussian']
    dice_g_val = history.history["val_dice_coef_gaussian"]
    iou_g_val = history.history["val_iou_coef_gaussian"]

    epochs = range(len(acc))

    plt.figure()
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training accuracy')
    plt.legend()
    acc_name = os.path.join(output_dir, "Accuracy_{}_{}.png".format(cfg.LABEL, datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
    plt.savefig(acc_name)
    plt.close()

    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training loss')
    plt.legend()
    loss_name = os.path.join(output_dir, "Loss_{}_{}.png".format(cfg.LABEL, datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
    plt.savefig(loss_name)
    plt.close()

    plt.figure()
    plt.plot(epochs, iou_g, 'r', label='IOU training')
    plt.plot(epochs, iou_g_val, 'g', label='IOU validation')
    plt.title('IoU')
    plt.legend()
    acc_name = os.path.join(output_dir, "IoU_{}_{}.png".format(cfg.LABEL, datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
    plt.savefig(acc_name)

    plt.figure()
    plt.plot(epochs, dice_g, 'r', label='F1 training')
    plt.plot(epochs, dice_g_val, 'g', label='F1 validation')
    plt.title('F1-score')
    plt.legend()
    acc_name = os.path.join(output_dir, "F1_{}_{}.png".format(cfg.LABEL, datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
    plt.savefig(acc_name)

