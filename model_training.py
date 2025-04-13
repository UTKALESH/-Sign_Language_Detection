import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import time

IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 50
TRAIN_DATA_DIR = 'data/train'
TEST_DATA_DIR = 'data/test'
MODEL_SAVE_PATH = 'saved_model/sign_language_model.keras'
NUM_CLASSES = len(os.listdir(TRAIN_DATA_DIR))
CLASS_NAMES = sorted(os.listdir(TRAIN_DATA_DIR))

print(f"Found {NUM_CLASSES} classes: {CLASS_NAMES}")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    color_mode='rgb'
)

print("Loading validation data...")
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    color_mode='rgb'
)

print("Loading test data...")
test_generator = test_datagen.flow_from_directory(
    TEST_DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=1,
    class_mode='categorical',
    shuffle=False,
    color_mode='rgb'
)

input_shape = (IMG_HEIGHT, IMG_WIDTH, 3 if train_generator.color_mode == 'rgb' else 1)
print(f"Input shape: {input_shape}")

print("Building the model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

print("Compiling the model...")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1,
                               restore_best_weights=True)

print("Starting training...")
start_time = time.time()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping]
)

training_time = time.time() - start_time
print(f"Training finished in {training_time:.2f} seconds.")

print("\nEvaluating on Test Data...")
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'\nTest Accuracy: {test_acc:.4f}')
print(f'Test Loss: {test_loss:.4f}')

if test_acc < 0.70:
     print("\nWARNING: Model accuracy is below the 70% minimum requirement!")
else:
     print("\nModel accuracy meets the minimum 70% requirement.")

print("\nGenerating Classification Report and Confusion Matrix...")

test_generator.reset()
Y_pred = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
y_pred_classes = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

if len(y_pred_classes) != len(y_true):
    print(f"Warning: Mismatch in prediction ({len(y_pred_classes)}) and true label ({len(y_true)}) counts. Truncating predictions.")
    y_pred_classes = y_pred_classes[:len(y_true)]

print('\nClassification Report:')
print(classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES))

print('\nConfusion Matrix:')
cm = confusion_matrix(y_true, y_pred_classes)
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved as confusion_matrix.png")

print(f"\nModel training complete. Best model saved to {MODEL_SAVE_PATH}")
print("Please ensure your dataset was sufficiently large and diverse for good real-world performance.")
