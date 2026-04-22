import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from tensorflow.keras.models import load_model

# Load the current .keras model
model = load_model('lstm_model.keras')

# Re-save as .h5 format
model.save('lstm_model.h5', save_format='h5')
print("Done! lstm_model.h5 created.")