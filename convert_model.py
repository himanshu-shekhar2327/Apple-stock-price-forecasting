import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from tensorflow.keras.models import load_model

# Load existing model
model = load_model('lstm_model.keras')

# Save as H5 format (version-tolerant)
model.save('lstm_model.h5')
print("Done! lstm_model.h5 created.")