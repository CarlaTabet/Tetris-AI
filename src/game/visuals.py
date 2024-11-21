import numpy as np
import cv2
import random


class VisualTetris:
    COLORS = [
        (0, 0, 0),  # Background
        (255, 105, 180),  # Pink
        (173, 216, 230),  # Light Blue
        (34, 139, 34),  # Green
        (255, 215, 0),  # Gold
        (138, 43, 226),  # Purple
        (255, 69, 0),  # Orange
        (0, 191, 255),  # Sky Blue
    ]

    SHAPES = [
        [[1, 1],
         [1, 1]],

        [[0, 2, 0],
         [2, 2, 2]],

        [[0, 3, 3],
         [3, 3, 0]],

        [[4, 4, 0],
         [0, 4, 4]],

        [[5, 5, 5, 5]],

        [[0, 0, 6],
         [6, 6, 6]],

        [[7, 0, 0],
         [7, 7, 7]],
    ]

    def __init__(self, height=20, width=10, block_size=30):
        self.height = height
        self.width = width
        self.block_size = block_size
        self.score = 0
        self.grid = np.zeros((height, width), dtype=int)
        self.active_piece = None
        self.piece_position = None
        self.game_over = False
        self.spawn_piece()

    def spawn_piece(self):
        """Spawns a new piece at the top of the board."""
        self.active_piece = random.choice(self.SHAPES)
        self.piece_position = {"x": self.width // 2 - len(self.active_piece[0]) // 2, "y": 0}

        # Check for game over condition
        if self.check_collision(self.piece_position["x"], self.piece_position["y"], self.active_piece):
            self.game_over = True

    def rotate_piece(self):
        """Rotates the active piece clockwise."""
        self.active_piece = [list(row) for row in zip(*self.active_piece[::-1])]

    def move_piece(self, dx, dy):
        """Moves the piece if the new position is valid."""
        new_x = self.piece_position["x"] + dx
        new_y = self.piece_position["y"] + dy

        if not self.check_collision(new_x, new_y, self.active_piece):
            self.piece_position = {"x": new_x, "y": new_y}
        elif dy > 0:  # Collision while moving down
            self.place_piece()

    def check_collision(self, x, y, piece):
        """Checks if a piece at a position collides with the board or edges."""
        for row_idx, row in enumerate(piece):
            for col_idx, cell in enumerate(row):
                if cell:
                    if (
                        y + row_idx >= self.height or
                        x + col_idx < 0 or
                        x + col_idx >= self.width or
                        (y + row_idx >= 0 and self.grid[y + row_idx][x + col_idx])
                    ):
                        return True
        return False

    def place_piece(self):
        """Places the active piece on the board and spawns a new one."""
        for row_idx, row in enumerate(self.active_piece):
            for col_idx, cell in enumerate(row):
                if cell:
                    self.grid[self.piece_position["y"] + row_idx][self.piece_position["x"] + col_idx] = cell
        self.clear_lines()
        self.spawn_piece()

    def clear_lines(self):
        """Clears completed lines and updates the score."""
        rows_to_clear = [i for i, row in enumerate(self.grid) if all(row)]
        for row in rows_to_clear:
            self.grid = np.delete(self.grid, row, axis=0)
            self.grid = np.vstack([np.zeros((1, self.width), dtype=int), self.grid])
        self.score += len(rows_to_clear) ** 2

    def render(self):
        """Renders the current game state with grid lines."""
        img = np.zeros((self.height * self.block_size, self.width * self.block_size + 200, 3), dtype=np.uint8)

        # Draw the grid
        for y in range(self.height):
            for x in range(self.width):
                color = self.COLORS[self.grid[y][x]]
                self.draw_block(img, x, y, color)

        # Draw the active piece
        if not self.game_over:
            for row_idx, row in enumerate(self.active_piece):
                for col_idx, cell in enumerate(row):
                    if cell:
                        x = self.piece_position["x"] + col_idx
                        y = self.piece_position["y"] + row_idx
                        self.draw_block(img, x, y, self.COLORS[cell])

        # Draw grid lines
        for x in range(0, self.width * self.block_size, self.block_size):
            cv2.line(img, (x, 0), (x, self.height * self.block_size), (50, 50, 50), 1)
        for y in range(0, self.height * self.block_size, self.block_size):
            cv2.line(img, (0, y), (self.width * self.block_size, y), (50, 50, 50), 1)

        # Overlay score and game-over message
        cv2.putText(img, f"Score: {self.score}", (self.width * self.block_size + 10, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)
        if self.game_over:
            cv2.putText(img, "GAME OVER", (10, self.height * self.block_size // 2),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3)

        # Display the game
        cv2.imshow("Classic Tetris", img)
        cv2.waitKey(1)

    def draw_block(self, img, x, y, color):
        """Draws a single block on the image."""
        x1, y1 = x * self.block_size, y * self.block_size
        x2, y2 = x1 + self.block_size, y1 + self.block_size
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)

    def step(self, action):
        """Takes a step in the game based on an action."""
        if not self.game_over:
            if action == "LEFT":
                self.move_piece(-1, 0)
            elif action == "RIGHT":
                self.move_piece(1, 0)
            elif action == "DOWN":
                self.move_piece(0, 1)
            elif action == "ROTATE":
                self.rotate_piece()


# Example Usage
if __name__ == "__main__":
    game = VisualTetris()
    while True:
        game.render()
        if game.game_over:
            cv2.waitKey(3000)  # Wait for 3 seconds before closing
            break
        game.step("DOWN")
    cv2.destroyAllWindows()
