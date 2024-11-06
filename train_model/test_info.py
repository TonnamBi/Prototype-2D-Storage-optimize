from tensorflow.keras.models import load_model

model_path = 'best_dqn_model.h5'
model = load_model(model_path, compile=False)

model.summary()

for layer in model.layers:
    try:
        weights, biases = layer.get_weights()
        print(f"\nLayer: {layer.name}")
        print("Weights:", weights)
        print("Biases:", biases)
    except ValueError:
     
        print(f"\nLayer: {layer.name} has no weights or biases.")
