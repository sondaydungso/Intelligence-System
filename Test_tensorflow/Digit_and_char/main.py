import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class EMNISTHandwritingModel:
    def __init__(self):
        self.num_classes = 62
        self.input_shape = (28, 28, 1)
        self.model = None
        self.class_names = self._create_class_names()

    def _create_class_names(self):
        class_names = []

        for i in range(10):
            class_names.append(str(i))
        
        for i in range(26):
            class_names.append(chr(ord('A') + i))

        for i in range(26):
            class_names.append(chr(ord('a') + i))

        return class_names
    
    def load_emnist_data(self):
        print("Loading EMNIST data...")

        (ds_train, ds_test), ds_info = tfds.load(
            'emnist/byclass',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        print(f"Number of training samples: {len(ds_train)}")
        print(f"Number of test samples: {len(ds_test)}")

        self.ds_train = ds_train
        self.ds_test = ds_test
        self.ds_info = ds_info
        
        return ds_train, ds_test, ds_info

    def preprocess_data(self, ds_train, ds_test):
        def normalize_img(image, label):
            # EMNIST data already has shape (28, 28, 1)
            # Apply transformations
            image = tf.image.rot90(image, k=3)
            image = tf.image.flip_left_right(image)
            image = tf.cast(image, tf.float32) / 255.0
            
            return image, label
        
        ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Create validation split
        train_size = ds_train.cardinality().numpy()
        val_size = int(0.2 * train_size)
        
        ds_val = ds_train.take(val_size)
        ds_train = ds_train.skip(val_size)
        
        # Optimize for performance
        BATCH_SIZE = 64  # Reduced batch size for better stability
        AUTOTUNE = tf.data.AUTOTUNE
        
        ds_train = ds_train.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
        ds_val = ds_val.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)
        ds_test = ds_test.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)
        
        return ds_train, ds_val, ds_test
    
    def create_model(self):
        return self._create_cnn_model()
        
    def _create_cnn_model(self):
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',  # Use sparse since labels are integers
            metrics=['accuracy']
        )

    def train_model(self, ds_train, ds_val, epochs=30):
        """Train the model with callbacks"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_emnist_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("Starting training...")
        history = self.model.fit(
            ds_train,
            epochs=epochs,
            validation_data=ds_val,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

    def evaluate_model(self, ds_test):
        """Evaluate model performance"""
        print("Evaluating model...")
        test_loss, test_accuracy = self.model.evaluate(ds_test, verbose=1)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Get predictions for detailed analysis
        y_true = []
        y_pred = []
        
        for images, labels in ds_test:
            predictions = self.model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(predictions, axis=1))
        
        return test_accuracy, y_true, y_pred
    def plot_confusion_matrix(self, y_true, y_pred, figsize=(15, 12)):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('EMNIST Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_samples(self, ds, num_samples=25):
        """Visualize sample images with predictions"""
        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
        axes = axes.ravel()
        
        # Get a batch of images
        images, labels = next(iter(ds.take(1)))
        
        for i in range(min(num_samples, len(images))):
            # Make prediction
            pred = self.model.predict(images[i:i+1], verbose=0)
            predicted_class = np.argmax(pred)
            confidence = np.max(pred)
            
            # Plot image
            axes[i].imshow(images[i].numpy().squeeze(), cmap='gray')
            axes[i].set_title(f'True: {self.class_names[labels[i]]}\n'
                            f'Pred: {self.class_names[predicted_class]} ({confidence:.2f})')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def predict_single_image(self, image):
        """Predict a single character from image"""
        if len(image.shape) == 2:
            image = image.reshape(1, 28, 28, 1)
        
        prediction = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return self.class_names[predicted_class], confidence
    

def train_emnist_model():
    """Complete training pipeline for EMNIST"""
    # Initialize model
    model = EMNISTHandwritingModel()
    
    # Load data
    ds_train, ds_test, ds_info = model.load_emnist_data()
    
    # Preprocess data
    ds_train, ds_val, ds_test = model.preprocess_data(ds_train, ds_test)
    
    # Create and compile model
    model.create_model()  # Create CNN model
    model.compile_model(learning_rate=0.001)    
    
    # Print model summary
    print("Model Architecture:")
    model.model.summary()
    
    # Train model
    history = model.train_model(ds_train, ds_val, epochs=10)  # Reduced epochs for testing
    
    # Plot training history
    model.plot_training_history(history)
    
    # Evaluate model
    test_accuracy, y_true, y_pred = model.evaluate_model(ds_test)
    
    # Visualize results
    model.visualize_samples(ds_test)
    
    # Plot confusion matrix (optional - can be large)
    # model.plot_confusion_matrix(y_true, y_pred)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=model.class_names))
    
    return model

if __name__ == "__main__":
    # Train the model
    trained_model = train_emnist_model()
    
    # Save the model
    trained_model.model.save('emnist_handwriting_model.keras')
    print("Model saved as 'emnist_handwriting_model.keras'")