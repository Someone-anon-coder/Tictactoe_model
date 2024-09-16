## TicTacToe AI

This project implements a TicTacToe AI using Q-learning. The AI agent learns to play optimally through reinforcement learning, aiming to maximize its chances of winning against an opponent. 

### Project Structure

The project is organized into the following folders and files:

* **agents:**
    * `agent1_q_table.pkl`: Saved Q-table for Player 1 (X)
    * `agent2_q_table.pkl`: Saved Q-table for Player 2 (O)
* **helper_classes:**
    * `environment.py`: Defines the TicTacToe game environment.
    * `q_learner.py`: Implements the Q-learning agent.
* **output_files:**
    * `actions_taken.txt`:  Stores the game state, available actions, and the chosen action for each move during a test game.
* **test_model.py:** Used to play against the trained AI agent.
* **train_model.py:** Trains the AI agent through Q-learning.

### Usage

**Training the AI:**

1. Run `train_model.py`.
2. The training process will iterate through a specified number of episodes.
3. The trained Q-tables for Player 1 and Player 2 will be saved in the `agents` folder.

**Testing the AI:**

1. Run `test_model.py`.
2. Choose whether you want to be Player 1 (X) or Player 2 (O).
3. The game will start, and the AI will make its moves based on the learned Q-table.
4. The game state, actions, and the AI's decision for each move will be saved in `actions_taken.txt`.

**Parameters:**

* **Episodes:** Number of training iterations.
* **Epsilon:** Exploration rate used during training.
* **Alpha:** Learning rate used during training.
* **Gamma:** Discount factor used during training.

**Requirements:**

* Python 3.x
* NumPy
* Pygame

### How it works

* **Environment:** The `environment.py` file defines the TicTacToe game environment. This includes functions for:
    * Initializing the game board.
    * Making a move.
    * Checking for a winner.
    * Rendering the game board visually.
* **Q-Learning Agent:** The `q_learner.py` file implements the Q-learning agent. This includes functions for:
    * Choosing an action based on the current state and Q-table.
    * Updating the Q-table based on the chosen action and its reward.
* **Training:** The `train_model.py` file runs the training process. It interacts with the environment and the Q-learning agent, allowing the agent to learn optimal strategies through repeated games.
* **Testing:** The `test_model.py` file allows you to play against the trained AI agent.

### Notes

* The AI is trained to win against a random opponent.
* The exploration rate (`epsilon`) is set to 0.25, allowing for some randomness in the agent's actions during training.
* The agent will learn to make moves that maximize its chances of winning, even in situations where it cannot guarantee a victory.
* The `actions_taken.txt` file can be used to analyze the AI's decision-making process during a test game.

**To improve the AI further:**

* Increase the number of training episodes.
* Tune the exploration rate, learning rate, and discount factor.
* Consider implementing more complex state representations or reward functions.
* Experiment with different Q-learning algorithms.

This project provides a basic foundation for building a TicTacToe AI using Q-learning. It demonstrates the potential of reinforcement learning to create intelligent agents that can achieve optimal performance in complex environments. 
