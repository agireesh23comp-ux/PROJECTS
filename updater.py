import tensorflow as tf

# Load the old model
model = tf.keras.models.load_model('tf-cnn-model.h5', compile=False)
print("Model loaded successfully!")

# Save it in the new format
model.save('tf-cnn-model-new.h5')
print("Model saved in new format!")