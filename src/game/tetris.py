import numpy as np
import pygame
import random


class Tetris:
    """
    Tetris game mechanics for reinforcement learning.
    """

    def __init__(self, height=20, width=10):
        """
        Initialize the Tetris environment.
        :param height: Number of rows in the grid.
        :param width: Number of columns in the grid.
        """
        self.height = height
        self.width = width
        self.grid = np.zeros((height, width), dtype=int)  # Grid initialized to empty
        self.score = 0
        self.state = "start"  # Game state: 'start' or 'gameover'
        self.figure = None
        self.colors = [
            (0, 0, 0),  # Empty cell
            (255, 105, 180),  # Pink
            (173, 216, 230),  # Light Blue
            (34, 139, 34),  # Green
            (255, 215, 0),  # Gold
            (138, 43, 226),  # Purple
            (255, 69, 0),  # Orange
        ]
        self.new_figure()

    def new_figure(self):
        """
        Spawns a new random Tetris piece.
        """
        self.figure = Figure(self.width // 2 - 2, 0)  # Center the piece horizontally

    def intersects(self):
        """
        Checks if the current piece intersects with the grid boundaries or blocks.
        :return: True if there is an intersection, False otherwise.
        """
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    x, y = self.figure.x + j, self.figure.y + i
                    if (
                        x < 0 or x >= self.width or  # Horizontal boundary check
                        y >= self.height or  # Vertical boundary check
                        (y >= 0 and self.grid[y][x] > 0)  # Block collision check
                    ):
                        return True
        return False

    def freeze(self):
        """
        Freezes the current piece on the grid, clears completed lines,
        and spawns a new piece. Ends the game if a new piece intersects.
        """
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    x, y = self.figure.x + j, self.figure.y + i
                    if y >= 0:
                        self.grid[y][x] = self.figure.color
        self.break_lines()
        self.new_figure()
        if self.intersects():
            self.state = "gameover"

    def break_lines(self):
        """
        Clears completed lines and updates the score.
        """
        lines = 0
        for i in range(self.height - 1, -1, -1):
            if all(self.grid[i]):  # Row is completely filled
                self.grid[1 : i + 1] = self.grid[:i]  # Shift rows down
                self.grid[0] = 0  # Clear the top row
                lines += 1
        self.score += lines ** 2  # Reward grows quadratically with cleared lines

    def move_piece(self, dx, dy):
        """
        Moves the piece horizontally or vertically.
        :param dx: Change in x (horizontal movement).
        :param dy: Change in y (vertical movement).
        """
        self.figure.x += dx
        self.figure.y += dy
        if self.intersects():
            self.figure.x -= dx
            self.figure.y -= dy
            if dy > 0:  # Collision when moving down
                self.freeze()

    def rotate_piece(self):
        """
        Rotates the current piece clockwise.
        """
        self.figure.rotate()
        if self.intersects():
            self.figure.rotation = (self.figure.rotation - 1) % len(self.figure.figures[self.figure.type])

    def get_board_state(self):
        """
        Returns the current grid as a binary NumPy array.
        :return: A 2D NumPy array of the grid.
        """
        state = self.grid.copy()
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    x, y = self.figure.x + j, self.figure.y + i
                    if 0 <= x < self.width and 0 <= y < self.height:
                        state[y][x] = self.figure.color
        return state

    def step(self, action):
        """
        Executes an action and updates the game state.
        :param action: Action to take ('LEFT', 'RIGHT', 'DOWN', 'ROTATE').
        :return: Tuple (next_state, reward, done).
        """
        if action == "LEFT":
            self.move_piece(-1, 0)
        elif action == "RIGHT":
            self.move_piece(1, 0)
        elif action == "DOWN":
            self.move_piece(0, 1)
        elif action == "ROTATE":
            self.rotate_piece()

        done = self.state == "gameover"
        reward = self.calculate_reward()
        next_state = self.get_board_state()
        return next_state, reward, done

    def calculate_reward(self):
        """
        Calculates the reward based on the current game state.
        :return: Reward for the current step.
        """
        if self.state == "gameover":
            return -100  # Heavy penalty for losing
        return self.score  # Reward based on cleared lines


class VisualTetris(Tetris):
    """
    Subclass of Tetris with rendering capabilities using pygame.
    """

    def __init__(self, height=20, width=10):
        super().__init__(height, width)
        pygame.init()
        self.screen_size = (width * 20 + 100, height * 20)
        self.block_size = 20
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()

    def render(self):
        """
        Renders the Tetris board and active piece using pygame.
        """
        self.screen.fill((255, 255, 255))  # White background

        # Draw grid
        for y in range(self.height):
            for x in range(self.width):
                color = self.colors[self.grid[y][x]]
                pygame.draw.rect(
                    self.screen,
                    color,
                    [x * self.block_size, y * self.block_size, self.block_size, self.block_size],
                )

        # Draw active piece
        if self.figure:
            for i in range(4):
                for j in range(4):
                    if i * 4 + j in self.figure.image():
                        x = self.figure.x + j
                        y = self.figure.y + i
                        if 0 <= x < self.width and 0 <= y < self.height:
                            pygame.draw.rect(
                                self.screen,
                                self.colors[self.figure.color],
                                [x * self.block_size, y * self.block_size, self.block_size, self.block_size],
                            )

        # Overlay score
        font = pygame.font.SysFont("Calibri", 25, True, False)
        text = font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(text, [10, 10])

        # Game over message
        if self.state == "gameover":
            font = pygame.font.SysFont("Calibri", 65, True, False)
            text_game_over = font.render("GAME OVER", True, (255, 0, 0))
            self.screen.blit(text_game_over, [50, self.height * self.block_size // 2])

        pygame.display.flip()

    def close(self):
        """
        Closes the pygame window.
        """
        pygame.quit()


class Figure:
    """
    Represents a Tetris piece.
    """

    figures = [
        [[1, 5, 9, 13], [4, 5, 6, 7]],
        [[4, 5, 9, 10], [2, 6, 5, 9]],
        [[6, 7, 9, 10], [1, 5, 6, 10]],
        [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
        [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],
        [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
        [[1, 2, 5, 6]],
    ]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = random.randint(0, len(self.figures) - 1)
        self.color = random.randint(1, 6)  # Assuming 6 colors
        self.rotation = 0

    def image(self):
        return self.figures[self.type][self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.figures[self.type])
