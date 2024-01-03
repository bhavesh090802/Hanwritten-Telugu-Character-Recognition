import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Constants
parent_folder_path = "D:/Online_classes/SEMESTER_7/Project/HWCR_project/trainingdata/"
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128  # Reduced image size
BATCH_SIZE = 32  # Reduced batch size

# Load and preprocess the Telugu character dataset from 52 folders
telugu_characters = []
labels = []

for folder_name in os.listdir(parent_folder_path):
    folder_path = os.path.join(parent_folder_path, folder_name)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                img_path = os.path.join(folder_path, filename)
                telugu_character = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                resized_character = cv2.resize(telugu_character, (IMAGE_WIDTH, IMAGE_HEIGHT))
                normalized_character = resized_character / 255.0  # Normalize pixel values
                telugu_characters.append(normalized_character)
                labels.append(folder_name)

telugu_characters = np.array(telugu_characters)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Define the number of classes (Telugu characters)
NUM_CLASSES = len(np.unique(encoded_labels))

# Reshape the data for model input (add channel dimension for grayscale images)
telugu_characters = np.reshape(telugu_characters, (telugu_characters.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1))

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(telugu_characters, encoded_labels, test_size=0.2, random_state=42)

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),   
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model using data generators with reduced batch size
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

# Train the model   
history = model.fit(train_dataset, epochs=30, validation_data=test_dataset)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict classes for test dataset
test_predictions = model.predict(test_dataset)
predicted_classes = np.argmax(test_predictions, axis=1)

# Decode labels back to original classes
decoded_predicted_classes = label_encoder.inverse_transform(predicted_classes)

# Recognition loop for Telugu characters
image_letter = 1
while os.path.isfile(f"telugu_characters/{image_letter}.png"):
    try:
        img = cv2.imread(f"telugu_characters/{image_letter}.png", cv2.IMREAD_GRAYSCALE)
        # Preprocess the image (resize, normalize, etc.)
        resized_img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        normalized_img = resized_img / 255.0
        img_array = np.reshape(normalized_img, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        predicted_telugu_character = label_encoder.classes_[predicted_class]  # Decode the class back to character
        print(f"This character is probably {predicted_telugu_character}")

        cv2.imshow('Character', resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

    finally:
        image_letter += 1

# Calculate confusion matrix
confusion_mtx = confusion_matrix(y_test, predicted_classes)

# Calculate classification report
class_report = classification_report(y_test, predicted_classes, target_names=label_encoder.classes_)

# Calculate precision, recall, and F1-score
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predicted_classes, average='weighted')

# Plot training & validation accuracy values
plt.figure(figsize=(15, 15))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(15, 15))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(NUM_CLASSES)
plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
plt.yticks(tick_marks, label_encoder.classes_)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("Confusion Matrix:")
print(confusion_mtx)
print("Classification Report:")
print(class_report)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")
