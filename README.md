
Here’s the updated summary with a bit more about Flask and the website:

"My project uses Deep Q-Learning, a type of reinforcement learning, combined with a greedy algorithm to optimize the placement of squares within a 30x30 cm area. I trained a DQN model in TensorFlow, which learned through rewards to place each square efficiently, maximizing filled space while avoiding overlap. The greedy algorithm helps prioritize placements to make the most out of available space.

Users can input up to 20 squares, and the model calculates the optimal layout. The project uses a Flask-based web application to make the AI accessible on a website. Flask handles requests from the user interface, passing square dimensions to the model and then returning the optimized placement. The results are displayed in real time with Matplotlib on the website, and any unplaced squares are shown as feedback. This setup demonstrates how reinforcement learning and greedy optimization work together, and how Flask facilitates interaction between the model and the user interface."
