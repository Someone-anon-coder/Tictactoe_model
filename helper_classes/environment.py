import time
import pygame
import numpy as np

class TicTacToe:
    def __init__(self) -> None:
        """
            Args: 
                None

            Returns:
                None

            Concept:
                Initializes class Tictactoe and sets up the board, current player, and player wins and pygame.
        """
        
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.player1_wincount = 0
        self.player2_wincount = 0
        self.draw_count = 0
        self.init_pygame()

    def init_pygame(self) -> None:
        """
            Args:
                None

            Returns:
                None

            Concept:
                Initializes pygame and sets up the screen, font, and clock.
        """
        
        pygame.init()
        self.screen = pygame.display.set_mode((300, 300))
        pygame.display.set_caption("Tic-Tac-Toe Training")
        self.font = pygame.font.Font(None, 74)
        self.clock = pygame.time.Clock()

    def reset(self) -> np.ndarray:
        """
            Args:
                None
            
            Returns:
                None
            
            Concept:
                Resets the board, current player, and player wins.
        """
        
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.render()
        
        return self.board

    def available_actions(self) -> list:
        """
            Args:
                None
            
            Returns:
                list of available actions
            
            Concept:
                Returns a list of all available actions for the agent on the board.
        """
        
        return list(zip(*np.where(self.board == 0)))
    
    def make_move(self, action: tuple[int, int]) -> tuple[int, bool]:
        """
            Args:
                action: Action selected by the agent (tuple)
            
            Returns:
                value of reward and whether the game is over with this move
                
            Concept:
                Makes a move on the board and updates the current player.
        """
        
        if self.board[action] == 0:
            self.board[action] = self.current_player
            reward, done = self.check_winner()
            self.render()
            self.current_player = -self.current_player
            
            return reward, done
        return 0, False
    
    def check_winner(self) -> tuple[int, bool]:
        """
            Args:
                None
            
            Returns:
                player who won and that the game is over
            
            Concept:
                Checks if there is a winner on the board.
        """
        
        for i in range(3):
            if np.all(self.board[i, :] == self.current_player) or np.all(self.board[:, i] == self.current_player):
                print(f"Player {1 if self.current_player == 1 else 2} wins!")
                print(self.board)
                
                if self.current_player == 1:
                    self.player1_wincount += 1
                else:
                    self.player2_wincount += 1
                
                return 1 if self.current_player == 1 else -1, True
        
        if np.all(np.diag(self.board) == self.current_player) or np.all(np.diag(np.fliplr(self.board)) == self.current_player):
            print(f"Player {1 if self.current_player == 1 else 2} wins!")
            print(self.board)
            
            if self.current_player == 1:
                self.player1_wincount += 1
            else:
                self.player2_wincount += 1
                
            return 1 if self.current_player == 1 else -1, True
        
        if not np.any(self.board == 0):
            print("It's a draw!")
            print(self.board)
            
            self.draw_count += 1
            
            return 0, True
        
        return 0, False

    def check_critical(self, player: int) -> tuple[float, tuple[int, int]]:
        """

        Args:
            player: specifier of player

        Returns:
            reward and critical position on the board and location of critical position.

        Concept:
            Checks if there is a critical position on the board. Critical position is a position that if one move can decide the winner.
        """
        
        # Checking if the current player has any critical position in in favor
        for i in range(3):
            if np.sum(self.board[i, :] == player) == 2 and np.any(self.board[i, :] == 0):
                return 0.1, (i, np.where(self.board[i, :] == 0)[0][0])  # Reward and location
            
            if np.sum(self.board[:, i] == player) == 2 and np.any(self.board[:, i] == 0):
                return 0.1, (np.where(self.board[:, i] == 0)[0][0], i)
        
        if np.sum(np.diag(self.board) == player) == 2 and np.any(np.diag(self.board) == 0):
            idx = np.where(np.diag(self.board) == 0)[0][0]
            return 0.1, (idx, idx)
        
        if np.sum(np.diag(np.fliplr(self.board)) == player) == 2 and np.any(np.diag(np.fliplr(self.board)) == 0):
            idx = np.where(np.diag(np.fliplr(self.board)) == 0)[0][0]
            return 0.1, (idx, 2 - idx)
        
        # Checking if the opponent has any critical position
        opponent = -player
        for i in range(3):
            if np.sum(self.board[i, :] == opponent) == 2 and np.any(self.board[i, :] == 0):
                return -0.1, (i, np.where(self.board[i, :] == 0)[0][0])
            
            if np.sum(self.board[:, i] == opponent) == 2 and np.any(self.board[:, i] == 0):
                return -0.1, (np.where(self.board[:, i] == 0)[0][0], i)
        
        if np.sum(np.diag(self.board) == opponent) == 2 and np.any(np.diag(self.board) == 0):
            idx = np.where(np.diag(self.board) == 0)[0][0]
            return -0.1, (idx, idx)
        
        if np.sum(np.diag(np.fliplr(self.board)) == opponent) == 2 and np.any(np.diag(np.fliplr(self.board)) == 0):
            idx = np.where(np.diag(np.fliplr(self.board)) == 0)[0][0]
            return -0.1, (idx, 2 - idx)
        
        return 0.2, None  # Default reward and no critical location

    def render(self) -> None:
        """
            Args:
                None
            
            Returns:
                None
            
            Concept:
                Renders the board on the screen.
        """
        
        self.screen.fill((255, 255, 255))
        for row in range(3):
            for col in range(3):
                if self.board[row, col] == 1:
                    pygame.draw.line(self.screen, (0, 0, 0), (col * 100 + 15, row * 100 + 15), (col * 100 + 85, row * 100 + 85), 15)
                    pygame.draw.line(self.screen, (0, 0, 0), (col * 100 + 15, row * 100 + 85), (col * 100 + 85, row * 100 + 15), 15)
                elif self.board[row, col] == -1:
                    pygame.draw.circle(self.screen, (0, 0, 0), (col * 100 + 50, row * 100 + 50), 40, 15)
        
        for i in range(1, 3):
            pygame.draw.line(self.screen, (0, 0, 0), (0, i * 100), (300, i * 100), 5)
            pygame.draw.line(self.screen, (0, 0, 0), (i * 100, 0), (i * 100, 300), 5)

        pygame.display.flip()
        
        time.sleep(0.1)  # Delay for 100 milliseconds # Uncomment to see the moves taken by the agent in real time

    def close_pygame(self) -> None:
        """
            Args:
                None
            
            Returns:
                None
            
            Concept:
                Closes the pygame window.
        """
        
        pygame.quit()