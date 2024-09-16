import pygame
from helper_classes.environment import TicTacToe
from helper_classes.q_learner import QLearningAgent

def train(episodes: int, epsilon: float, alpha: float, gamma: float) -> None:
    env = TicTacToe()
    agent1 = QLearningAgent(player=1, epsilon=epsilon, alpha=alpha, gamma=gamma)
    agent2 = QLearningAgent(player=-1, epsilon=epsilon, alpha=alpha, gamma=gamma)
    
    try:
        for episode in range(episodes):
            print(f"Episode {episode + 1}/{episodes}")
            
            state = env.reset()
            done = False
            while not done:
                # print(env.board)
                player = env.current_player
                agent = agent1 if player == 1 else agent2
                state_tuple = agent.get_state(state)
                actions = env.available_actions()
                action = agent.choose_action(state, actions)
                
                reward, critical_location = env.check_critical(player) # Check critical conditions
                
                """
                    If critical location is given and agent wins, give 2 reward points
                    If critical location is given and agent doesn't mark there and loses, give 2 penalty points
                """
                if critical_location:
                    if action == critical_location:
                        reward = 2
                    elif done and reward == -1:
                        reward = -2
                
                reward_move, done = env.make_move(action)
                next_state_tuple = agent.get_state(env.board)
                next_actions = env.available_actions()
                agent.update_q_table(state_tuple, action, reward + reward_move, next_state_tuple, next_actions)
                
                # Update Q table for the other agent in case of losing
                if done and reward != 0:
                    other_agent = agent2 if player == 1 else agent1
                    other_agent.update_q_table(state_tuple, action, -reward, next_state_tuple, next_actions)
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

    finally:
        env.close_pygame()
        print(f"Player 1 wins: {env.player1_wincount} times")
        print(f"Player 2 wins: {env.player2_wincount} times")
        print(f"Draws: {env.draw_count}")

    agent1.save_model('agents\\agent1_q_table.pkl')
    agent2.save_model('agents\\agent2_q_table.pkl')

if __name__ == "__main__":
    epsilon = 0.25    # Exploration Rate
    alpha = 0.07      # Learning Rate
    gamma = 0.8       # Discount Factor
    episodes = 100000 # Iteration Count
    
    train(episodes=episodes, epsilon=epsilon, alpha=alpha, gamma=gamma)