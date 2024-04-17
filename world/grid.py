"""Grid.

Credit to Tom v. Meer for writing this.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path


class Grid:
    def __init__(self, n_cols: int, n_rows: int):
        """Grid representation of the world 
           as a 2D numpy integer array.

        Possible grid values are:
        - Empty: 0,
        - Boundary: 1,
        - Obstacle: 2,
        - Dirt: 3,
        - Charger: 4

        Args:
            n_cols: Number of grid columns.
            n_rows: Number of grid rows.
        """

        # Building the boundary of the grid:
        self.cells = np.zeros((n_cols, n_rows), dtype=np.int8)
        self.cells[0, :] = self.cells[-1, :] = 1
        self.cells[:, 0] = self.cells[:, -1] = 1
        
        self.n_rows = self.cells.shape[1]
        self.n_cols = self.cells.shape[0]

        self.objects = {
            "empty": 0,
            "boundary": 1,
            "obstacle": 2,
            "target": 3,
            "charger": 4
        }

    def place_object(self, x, y, type):
        """Places an object on the grid.

        Args:
            x: x-coordinate of the object.
            y: y-coordinate of the object.
            type: Type of the object.
        """
        self.cells[x][y] = self.objects[type]

    @staticmethod
    def load_grid(fp: Path) -> Grid:
        """Loads a numpy array from file path.

        Returns:
            A Grid object from the file.
        """
        arr = np.load(fp)
        g = Grid(arr.shape[0], arr.shape[1])
        g.cells = arr

        return g

    def save_grid_file(self, fp: Path):
        """Saves the numpy array representation of 
        the grid to file path.

        Args:
            fp: File path where the grid file is to be saved.
        """
        np.save(fp.with_suffix(".npy"), self.cells)