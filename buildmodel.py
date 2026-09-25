from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

imgWidth = 256
imgHeight = 256 
batchSize = 32 
epochs = 25  # adjust if needed

def build_and_train_model(train_dir, val_dir, label):
    datagen = ImageDataGenerator(
        rescale=1/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        zoom_range=0.3,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1/255.0)

    train_data = datagen.flow_from_directory(train_dir,
                                             batch_size=batchSize,
                                             class_mode='categorical',
                                             target_size=(imgHeight, imgWidth))

    val_data = val_datagen.flow_from_directory(val_dir,
                                               batch_size=batchSize,
                                               class_mode='categorical',
                                               target_size=(imgHeight, imgWidth))

    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(imgHeight, imgWidth, 3)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(256, (3, 3), activation='relu'),
        Conv2D(256, (3, 3), activation='relu'),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(5, activation='softmax')
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    callback = EarlyStopping(monitor="val_loss", patience=5, verbose=1, mode='auto')
    checkpoint = ModelCheckpoint(f"bestModel_{label}.h5", monitor="val_accuracy", verbose=1, save_best_only=True)

    history = model.fit(train_data,
                        epochs=epochs,
                        verbose=1,
                        validation_data=val_data,
                        callbacks=[callback, checkpoint])
    
    return history

# Define paths
orig_train = "C:/Users/Shreyas/Desktop/Weather_ANN/dataset/Train"
orig_val   = "C:/Users/Shreyas/Desktop/Weather_ANN/dataset/Validate"

filt_train = "C:/Users/Shreyas/Desktop/Weather_ANN/dataset_filtered/Train"
filt_val   = "C:/Users/Shreyas/Desktop/Weather_ANN/dataset_filtered/Validate"

# Train both models
history_orig = build_and_train_model(orig_train, orig_val, "Original")
history_filt = build_and_train_model(filt_train, filt_val, "Filtered")

# Plotting
def plot_histories(hist1, hist2, label1="Original", label2="Filtered"):
    epochs_range = range(len(hist1.history['accuracy']))

    plt.figure(figsize=(14, 6))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, hist1.history['accuracy'], 'r-', label=f'{label1} Train')
    plt.plot(epochs_range, hist1.history['val_accuracy'], 'r--', label=f'{label1} Val')
    plt.plot(epochs_range, hist2.history['accuracy'], 'b-', label=f'{label2} Train')
    plt.plot(epochs_range, hist2.history['val_accuracy'], 'b--', label=f'{label2} Val')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, hist1.history['loss'], 'r-', label=f'{label1} Train')
    plt.plot(epochs_range, hist1.history['val_loss'], 'r--', label=f'{label1} Val')
    plt.plot(epochs_range, hist2.history['loss'], 'b-', label=f'{label2} Train')
    plt.plot(epochs_range, hist2.history['val_loss'], 'b--', label=f'{label2} Val')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_histories(history_orig, history_filt)
