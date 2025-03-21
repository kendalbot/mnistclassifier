import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

print("Loading dataset...")
(x_train, y_train) (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(f"Dataset Training: {x_train.shape}, Test: {x_test.shape}")


x_train = x_train / 255.0  
x_test = x_test / 255.0



#Reshape from 28x28 to flat 784-pixel, vector
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)




#Sample Imgs
plt.figure(figsize=(12, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('sample_images.png')
plt.close()







print("Building the neural network model...")
model = tf.keras.Sequential([
    #Input Layer to hidden 1
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    
    #Hl1 to hl2
    tf.keras.layers.Dense(64, activation='relu'),
    
    #Hidden 2 Output layer (10 digits)
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()





#c
print("Compiling the model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)





#t
print("Training the model...")
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)


#e
print("Evaluating...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

#p
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('learning_curves.png')
plt.close()
#predictions on test data
predictions = model.predict(x_test)
y_pred_classes = np.argmax(predictions, axis=1)
#Confusion matrixcompute
#Confusion matrix plot
cm = confusion_matrix
