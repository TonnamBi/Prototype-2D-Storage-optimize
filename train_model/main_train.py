import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from collections import deque
import os

class StorageEnvironment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.width, self.height))
        self.free_space = self.width * self.height
        return self.grid

    def render(self, episode, current_test, total_tests, free_space):
        plt.imshow(self.grid, cmap='Greys', origin='upper')
        plt.colorbar(label='Occupancy')
        plt.title(f"Storage Area - Episode {episode} | Test {current_test}/{total_tests} | Free Space: {free_space}")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.show()

    def place_square(self, square_width, square_height, greedy=False):
        if greedy:
            return self.greedy_place_square(square_width, square_height)
        else:
            for x in range(self.width - square_width + 1):
                for y in range(self.height - square_height + 1):
                    if np.sum(self.grid[x:x+square_width, y:y+square_height]) == 0:
                        self.grid[x:x+square_width, y:y+square_height] = 1
                        self.free_space -= square_width * square_height
                        reward = self._calculate_reward(square_width, square_height)
                        return reward, True
            return -0.1, False

    def greedy_place_square(self, square_width, square_height):
        best_position = None
        min_wasted_space = self.free_space

        for x in range(self.width - square_width + 1):
            for y in range(self.height - square_height + 1):
                if np.sum(self.grid[x:x+square_width, y:y+square_height]) == 0:
                    wasted_space = self._calculate_wasted_space(x, y, square_width, square_height)
                    if wasted_space < min_wasted_space:
                        min_wasted_space = wasted_space
                        best_position = (x, y)

        if best_position:
            x, y = best_position
            self.grid[x:x+square_width, y:y+square_height] = 1
            self.free_space -= square_width * square_height
            reward = self._calculate_reward(square_width, square_height)
            return reward, True
        return -0.1, False

    def _calculate_wasted_space(self, x, y, square_width, square_height):
        free_cells = np.sum(self.grid == 0)
        projected_free = np.sum(self.grid[x:x+square_width, y:y+square_height] == 0)
        return free_cells - projected_free

    def _calculate_reward(self, square_width, square_height):
        size_factor = (square_width * square_height) / (self.width * self.height)
        return size_factor * (self.free_space / (self.width * self.height + 1e-9))

class DQNAgent:
    def __init__(self, state_shape, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, target_update=10):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.target_update = target_update
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.state_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_target_network(self, episode):
        if episode % self.target_update == 0:
            self.update_target_model()

    def save(self, model_path="dqn_storage_model.h5"):
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def load(self, model_path="dqn_storage_model.h5"):
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print("Model file not found.")

def train_agent(agent, environment, episodes=5000, batch_size=32, log_dir="logs"):
    log_path = os.path.join(log_dir, "training_logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)

    max_reward = -float('inf')
    patience = 50
    no_improvement = 0

    for e in range(episodes):
        state = environment.reset().reshape(1, *environment.grid.shape)
        total_reward = 0
        for time in range(500):
            action = agent.act(state)
            x = (action % environment.width)
            y = (action // environment.width)
            square_width = random.randint(1, 20)
            square_height = random.randint(1, 20)
            reward, done = environment.place_square(square_width, square_height, greedy=(e % 2 == 0))
            next_state = environment.grid.reshape(1, *environment.grid.shape)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        agent.train_target_network(e)

        if total_reward > max_reward:
            max_reward = total_reward
            agent.save("best_dqn_model.h5")
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f"Early stopping at episode {e + 1}")
            break

        print(f"Episode {e + 1}/{episodes}, Total reward: {total_reward}")

def test_model(agent, environment, num_tests=10):
    total_filled_area = 0
    total_free_space = 0
    total_reward = 0
    total_successful_placements = 0
    total_placement_attempts = 0

    print("\nStarting Enhanced Greedy Optimization Testing...")

    for test in range(num_tests):
        state = environment.reset().reshape(1, *environment.grid.shape)
        test_reward = 0
        num_squares = random.randint(1, 20)  
        print(f"\nTest {test + 1}/{num_tests}: Attempting to place {num_squares} squares")
        
        successful_placements = 0
        for _ in range(num_squares):
            square_width, square_height = random.randint(1, 20), random.randint(1, 20)
            reward, done = environment.place_square(square_width, square_height, greedy=True)
            test_reward += reward
            total_placement_attempts += 1
            if done:
                successful_placements += 1
                total_successful_placements += 1
            else:
                break

        total_filled_area += (environment.width * environment.height) - environment.free_space
        total_free_space += environment.free_space
        total_reward += test_reward
        environment.render(test + 1, test + 1, num_tests, environment.free_space)

        print(f"Test {test + 1}/{num_tests} - Successful Placements: {successful_placements}/{num_squares}, "
              f"Reward: {test_reward}, Remaining Free Space: {environment.free_space}")

    avg_filled_area = total_filled_area / num_tests
    avg_free_space = total_free_space / num_tests
    avg_reward = total_reward / num_tests
    success_rate = (total_successful_placements / total_placement_attempts) * 100

    print("\nEnhanced Greedy Optimization Test Summary:")
    print(f"Average Filled Area: {avg_filled_area:.2f}")
    print(f"Average Free Space: {avg_free_space:.2f}")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Placement Success Rate: {success_rate:.2f}%")

environment = StorageEnvironment(30, 30)
state_shape = environment.grid.shape
agent = DQNAgent(state_shape=state_shape, action_size=environment.width * environment.height)
train_agent(agent, environment, episodes=500, batch_size=32)
test_model(agent, environment)
