from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('./model.h5')

# Convert the model to JSON format
model_json = model.to_json()

# Save the model JSON to a file
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
