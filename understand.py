from tensorflow.keras.models import load_model

model = load_model("converted_keras/keras_model.h5")

model.summary()