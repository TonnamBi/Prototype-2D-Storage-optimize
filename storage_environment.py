import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class StorageEnvironment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.square_colors = []  
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.width, self.height))
        self.free_space = self.width * self.height
        self.square_colors.clear()
        return self.grid

    def render(self, free_space=0):
        unique_squares = np.unique(self.grid)
        cmap = plt.cm.get_cmap("tab20", len(unique_squares))
        norm = mcolors.BoundaryNorm(boundaries=range(len(unique_squares) + 1), ncolors=len(unique_squares))
        
        plt.imshow(self.grid, cmap=cmap, norm=norm, origin='upper')
        plt.colorbar(ticks=range(len(unique_squares)), label='Square ID')
        plt.title(f"Storage Area - Free Space: {free_space}")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.show()

    def place_square(self, square_width, square_height, greedy=False):
        square_id = len(self.square_colors) + 1
        self.square_colors.append(np.random.rand(3,))  
        
        if greedy:
            return self.greedy_place_square(square_width, square_height, square_id)
        else:
            for x in range(self.width - square_width + 1):
                for y in range(self.height - square_height + 1):
                    if np.sum(self.grid[x:x+square_width, y:y+square_height]) == 0:
                        self.grid[x:x+square_width, y:y+square_height] = square_id
                        self.free_space -= square_width * square_height
                        return 1.0, True
            return -0.1, False

    def greedy_place_square(self, square_width, square_height, square_id):
        best_position = None
        min_wasted_space = self.free_space

        for x in range(self.width - square_width + 1):
            for y in range(self.height - square_height + 1):
                if np.sum(self.grid[x:x+square_width, y:y+square_height]) == 0:
                    wasted_space = np.sum(self.grid == 0) - np.sum(self.grid[x:x+square_width, y:y+square_height] == 0)
                    if wasted_space < min_wasted_space:
                        min_wasted_space = wasted_space
                        best_position = (x, y)

        if best_position:
            x, y = best_position
            self.grid[x:x+square_width, y:y+square_height] = square_id
            self.free_space -= square_width * square_height
            return 1.0, True
        return -0.1, False
