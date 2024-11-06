from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from tensorflow.keras.models import load_model
from storage_environment import StorageEnvironment
import matplotlib.colors as mcolors

app = Flask(__name__)
model = load_model("best_dqn_model.h5")
environment = StorageEnvironment(30, 30)

def generate_display_image(grid, free_space=0):
    plt.figure(figsize=(6, 6))
    
    unique_squares = np.unique(grid)
    colors = ["white"] + [plt.cm.tab20(i) for i in range(1, len(unique_squares))]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries=range(len(unique_squares) + 1), ncolors=len(unique_squares))

    plt.imshow(grid, cmap=cmap, norm=norm, origin='upper')
    plt.colorbar(ticks=range(len(unique_squares)), label="Square ID")
    plt.title(f"AI Optimized Storage - Free Space: {free_space}")
    plt.xlabel("Width")
    plt.ylabel("Height")

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json.get('squares', [])
    if not data:
        return jsonify({"error": "No squares provided."}), 400

    environment.reset()

    for square in data:
        square_width, square_height = square.get('width'), square.get('height')
        reward, placed = environment.place_square(square_width, square_height, greedy=True)
        if not placed:
            break

    image = generate_display_image(environment.grid, free_space=environment.free_space)
    return send_file(image, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
