import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, player: int, epsilon: int=0.1, alpha: int=0.5, gamma: int=0.9) -> None:
        """
            Args:
                player: The specifier for the current player
                epsilon: The exploration rate
                alpha: The learning rate
                gamma: The discount factor
            
            Returns:
                None
            
            Concept:
                Initializes the class QLearningAgent with the given parameters; and sets the q_table to an empty dictionary.
        """
        
        self.q_table = defaultdict(lambda: 0)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.player = player
    
    def get_state(self, board: np.ndarray) -> tuple:
        """
            Args:
                board: The current board being played at the moment
            
            Returns:
                State as the board condition
                
            Concept:
                Returns the state as the board condition.
        """
        
        return tuple(map(tuple, board))
    
    def choose_action(self, board: np.ndarray, available_actions: list) -> int:
        """
            Args:
                board: The current board being played at the moment
                available_actions: A list of actions that can be taken on the current board
            
            Returns:
                A calculated action of the available actions
            
            Concept:
                Returns a calculated action of the available actions and if epsilon value is greater than a random value then a random action is returned.
        """
        
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        
        q_values = [self.q_table[(self.get_state(board), action)] for action in available_actions]
        max_q = max(q_values)
        
        return random.choice([a for a, q in zip(available_actions, q_values) if q == max_q])
    
    def update_q_table(self, state: tuple, action: int, reward: int, next_state: tuple, next_actions: list) -> None:
        """
            Args:
                state: The current board at the moment
                action: Action taken by the agent
                reward: reward given for that action in that condition
                next_state: The next board to played
                next_actions: Actions available to model on the next board
            
            Returns:
                None
            
            Concept:
                Updates the q_table with the given parameters.
        """
        
        current_q = self.q_table[(state, action)]
        if next_actions:
            max_next_q = max([self.q_table[(next_state, a)] for a in next_actions])
        else:
            max_next_q = 0
        self.q_table[(state, action)] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
    
    def save_model(self, filename: str) -> None:
        """
            Args:
                filename: filename for the q_table to be saved in
            
            Returns:
                None
            
            Concept:
                Saves the q_table in the given filename.
        """
        
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_model(self, filename: str) -> None:
        """
            Args:
                filename: filename for the q_table to be loaded from
            
            Returns:
                None
            
            Concept:
                Loads the q_table from the given filename.
        """
        
        import pickle
        with open(filename, 'rb') as f:
            self.q_table = defaultdict(lambda: 0, pickle.load(f))