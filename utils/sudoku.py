import numpy as np
from .solver import Backtracking, DualAffineSolver


class Sudoku(object):
    def __init__(self, clues):
        self.cells = clues

    @property
    def cells(self):
        return self._cells

    @cells.setter
    def cells(self, cells):
        if not isinstance(cells, np.ndarray):
            raise TypeError("Sudoku: clues must be of type np.ndarray")
        if (np.any(cells > 9)) or (np.any(cells < 0)):
            raise ValueError("Sudoku: clues must be between 0 and 9 (0 and 9 including)")
        self._cells = cells


    def solve(self, method="DA", plot_matrices=False, max_iterations=1e10, iterations=5):
        bt_solver = Backtracking(self)
        if method == "DA":
            da_solver = DualAffineSolver(self)
            da_solver.iterations = iterations
            da_solver.solve()
            if plot_matrices:
                da_solver.plot_matrices()
                da_solver.plot_objective_function_values()
                
            self.title = "Dual Affine Scaling"
        elif method == "backtrack":
            #bt_solver = Backtracking(self)
            bt_solver.max_iterations = max_iterations
            bt_solver.solve()
            self.title = "Backtracking"
        else:
            raise ValueError(f"Solving method {method} is not implemented.\
Following methods are available:\nDA: Dual Affine Scaling\nbacktrack: Backtracking")
        #if bt_solver.check_board(self.cells):
        #    print("Looks good to me ;)")
