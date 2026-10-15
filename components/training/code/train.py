import argparse
import os
from glob import glob
import random
import tensorflow as tf
import numpy as np

# TensorFlow / Keras libraries for AI training
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Azure ML package for logging metrics, etc.
from azureml.core import Run

# Utility functions from utils.py
from utils import *


def main():

    # ---------------------------------------------------------
    # ARGUMENTS
    # ---------------------------------------------------------

    parser = argparse.ArgumentParser()

    # Existing arguments
    parser.add_argument(
        "--training_folder",
        type=str,
        dest="training_folder",
        help="Training folder mounting point"
    )

    parser.add_argument(
        "--testing_folder",
        type=str,
        dest="testing_folder",
        help="Testing folder mounting point"
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        dest="output_folder",
        help="Output folder"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        dest="epochs",
        help="The amount of epochs to train"
    )

    # ---------------------------------------------------------
    # NEW DYNAMIC PARAMETERS
    # ---------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--initial_learning_rate",
        type=float,
        default=0.01,
        help="Initial learning rate for the optimizer"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size used during training"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=11,
        help="Number of epochs to wait before early stopping"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="animal-cnn",
        help="Name used to save the trained model"
    )

    args = parser.parse_args()

    # Print all arguments
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    # ---------------------------------------------------------
    # INPUT / OUTPUT PATHS
    # ---------------------------------------------------------

    training_folder = args.training_folder
    print("Training folder:", training_folder)

    testing_folder = args.testing_folder
    print("Testing folder:", testing_folder)

    output_folder = args.output_folder
    print("Output folder:", output_folder)

    MAX_EPOCHS = args.epochs

    # ---------------------------------------------------------
    # DYNAMIC TRAINING SETTINGS
    # ---------------------------------------------------------

    SEED = args.seed
    INITIAL_LEARNING_RATE = args.initial_learning_rate
    BATCH_SIZE = args.batch_size
    PATIENCE = args.patience
    model_name = args.model_name

    print("Seed:", SEED)
    print("Initial learning rate:", INITIAL_LEARNING_RATE)
    print("Batch size:", BATCH_SIZE)
    print("Patience:", PATIENCE)
    print("Model name:", model_name)

    # ---------------------------------------------------------
    # LOAD TRAINING / TESTING FILE PATHS
    # ---------------------------------------------------------

    training_paths = glob(
        training_folder + "/*.jpg",
        recursive=True
    )

    testing_paths = glob(
        testing_folder + "/*.jpg",
        recursive=True
    )

    print("Training samples:", len(training_paths))
    print("Testing samples:", len(testing_paths))

    # ---------------------------------------------------------
    # SHUFFLE DATA
    # ---------------------------------------------------------

    random.seed(SEED)
    random.shuffle(training_paths)

    random.seed(SEED)
    random.shuffle(testing_paths)

    print(training_paths[:3])
    print(testing_paths[:3])

    # ---------------------------------------------------------
    # LOAD FEATURES AND TARGETS
    # ---------------------------------------------------------

    X_train = getFeatures(training_paths)
    y_train = getTargets(training_paths)

    X_test = getFeatures(testing_paths)
    y_test = getTargets(testing_paths)

    print("Shapes:")
    print(X_train.shape)
    print(X_test.shape)
    print(len(y_train))
    print(len(y_test))

    # ---------------------------------------------------------
    # ENCODE LABELS
    # ---------------------------------------------------------

    LABELS, y_train, y_test = encodeLabels(
        y_train,
        y_test
    )

    print("One Hot Shapes:")
    print(y_train.shape)
    print(y_test.shape)

    # ---------------------------------------------------------
    # OUTPUT LOCATION FOR MODEL
    # ---------------------------------------------------------

    model_directory = os.path.join(
        output_folder,
        model_name
    )

    os.makedirs(
        model_directory,
        exist_ok=True
    )

    model_path = os.path.join(
        model_directory,
        "model.keras"
    )

    print(f"model_path: {model_path}")

    # ---------------------------------------------------------
    # CALLBACKS
    # ---------------------------------------------------------

    # Save only the best model
    cb_save_best_model = keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # Stop training when validation loss stops improving
    cb_early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        verbose=1,
        restore_best_weights=True
    )

    # Reduce learning rate when validation loss stops improving
    cb_reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=4,
        verbose=1
    )

    # ---------------------------------------------------------
    # OPTIMIZER
    # ---------------------------------------------------------

    opt = SGD(
        learning_rate=INITIAL_LEARNING_RATE
    )

    # ---------------------------------------------------------
    # BUILD MODEL
    # ---------------------------------------------------------

    model = buildModel(
        (64, 64, 3),
        3
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )

    # ---------------------------------------------------------
    # DATA AUGMENTATION
    # ---------------------------------------------------------

    aug = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # ---------------------------------------------------------
    # TRAIN MODEL
    # ---------------------------------------------------------

    history = model.fit(
        aug.flow(
            X_train,
            y_train,
            batch_size=BATCH_SIZE
        ),
        validation_data=(
            X_test,
            y_test
        ),
        steps_per_epoch=max(
            1,
            len(X_train) // BATCH_SIZE
        ),
        epochs=MAX_EPOCHS,
        callbacks=[
            cb_save_best_model,
            cb_early_stop,
            cb_reduce_lr_on_plateau
        ]
    )

    # ---------------------------------------------------------
    # EVALUATE MODEL
    # ---------------------------------------------------------

    print("[INFO] evaluating network...")

    predictions = model.predict(
        X_test,
        batch_size=32
    )

    print(
        classification_report(
            y_test.argmax(axis=1),
            predictions.argmax(axis=1),
            target_names=[
                "cats",
                "dogs",
                "panda"
            ]
        )
    )

    # ---------------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------------

    cf_matrix = confusion_matrix(
        y_test.argmax(axis=1),
        predictions.argmax(axis=1)
    )

    print(cf_matrix)

    # Save the confusion matrix to the output folder
    np.save(
        os.path.join(
            output_folder,
            "confusion_matrix.npy"
        ),
        cf_matrix
    )

    print("DONE TRAINING")


if __name__ == "__main__":
    main()