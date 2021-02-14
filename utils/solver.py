import scipy.optimize
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Dual Affine Scaling Solver Class
class DualAffineSolver(object):
    def __init__(self, sudoku):
        self.sudoku = sudoku
        self.iterations = 5

        self._clues = self.sudoku.given_clues
        self._objective_function_values = []
        

    def solve(self):
        c = np.ones((self.A.shape[1]))[:,None]
        b = np.ones((self.A.shape[0]))[:,None]
        y = self.solve_dual_affine_scaling(self.A, b, c)
        self.solve_sudoku(self.A, y, c)

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        if epsilon > 0:
            self._epsilon = epsilon
        else:
            raise ValueError("Epsilon cannot be samller or equal to 0!")

    @property
    def iterations(self):
        return self._iterations
    
    @iterations.setter
    def iterations(self, iterations):
        if iterations > 1:
            self._iterations = iterations
        else:
            raise ValueError("Iterations must be bigger than 1")

    @property
    def A(self):
        return sp.vstack([self.A_row, self.A_col, self.A_box, self.A_color, self.A_cell, self.A_clue])

    @property
    def A_row(self):
        # row constraints
        I9 = sp.eye(9)
        Ir = sp.bmat([[I9, I9, I9, I9, I9, I9, I9, I9, I9]])
        A_row = sp.block_diag([Ir, Ir, Ir, Ir, Ir, Ir, Ir, Ir, Ir])
        return A_row

    @property
    def A_col(self):
        # column constraints
        I81 = sp.eye(81)
        A_col = sp.bmat([[I81, I81, I81, I81, I81, I81, I81, I81, I81]])
        
        return A_col

    @property
    def A_box(self):
        # box constraints
        I9 = sp.eye(9)
        I9_27 = sp.bmat([[I9, I9, I9]])
        Ab = sp.block_diag([I9_27, I9_27, I9_27])
        Ab1 = sp.bmat([[Ab, Ab, Ab]])
        A_box = sp.block_diag([Ab1, Ab1, Ab1])
        return A_box

    @property
    def A_color(self):
        I27 = sp.eye(27)
        Ac = sp.bmat([[I27, I27, I27]])
        Ac1 = sp.block_diag([Ac, Ac, Ac])
        A_color = sp.bmat([[Ac1, Ac1, Ac1]])
        return A_color

    @property
    def A_cell(self):
        # cell constraints
        ones_9 = np.ones((1,9))
        Ic = sp.block_diag([ones_9, ones_9, ones_9, ones_9, ones_9, ones_9, ones_9, ones_9, ones_9])
        A_cell = sp.block_diag([Ic, Ic, Ic, Ic, Ic, Ic, Ic, Ic, Ic])
        return A_cell
    
    @property
    def A_clue(self):
        clues_onehot = np.zeros((self._clues[:,2].size, 9))
        clues_onehot[np.arange(self._clues[:,2].size),self._clues[:,2] - 1] = 1

        clues_onehot_index = np.asarray([self._clues[:,0] - 1, self._clues[:,1] - 1]).T

        pad_before_index = np.asarray([clues_onehot_index[:,0]*81 + clues_onehot_index[:,1]*9]).T
        pading_indices = np.hstack((pad_before_index[:,:], 720 - pad_before_index[:,:])) #729 - 9 - pad before => pad_before + 9 + pad_after = 729
        
        # Not soooo a cool vectorized solution ...
        A_clue = []
        for pad_indices, onehot_clue in zip(pading_indices, clues_onehot):
            padded_a = np.pad(onehot_clue, pad_indices, "constant", constant_values=(0))

            A_clue.append(padded_a)
        A_clue = np.asarray(A_clue)

        return A_clue
    
    def solve_dual_affine_scaling(self, A, b, c):
        y = np.zeros((b.shape))
        c = c
        self.epsilon = 0.01
        #print(f"\nA: {A.shape}\nb: {b.shape}\nc: {c.shape}\ny: {y.shape}")
        pref_objective_function_value = 0
        for k in range(self.iterations):
            print(f"Dual affine scaling Iteration {k+1}/{self.iterations}", end="\r")
            s_k = c - A.T@y
            H_k = np.diag(1/(s_k[:,0]**2))
            H_k_inverse = np.linalg.pinv(A@H_k@A.T)@b

            d_s = (-A.T@H_k_inverse) 
            
            t = 0.9*(-s_k/(d_s))[d_s<0].min()
            y = y + t*(H_k_inverse)

            objective_function_value = (b.T@y).flatten()[0]
            self._objective_function_values.append(objective_function_value)
            if objective_function_value < pref_objective_function_value + self.epsilon:
                break
            pref_objective_function_value = objective_function_value
        return y

    def plot_objective_function_values(self):
        num_iters = len(self._objective_function_values) + 1
        title = f"Values of objective Function for {num_iters - 1} iterations"
        fig, ax = plt.subplots()
        plt.grid()
        ax.plot(range(1,num_iters), self._objective_function_values, color="springgreen")

        if num_iters <= 12: # Add Annotations only if there are a few iterations
            ax.scatter(range(1,num_iters), self._objective_function_values, color="cyan")
            for i, value in enumerate(self._objective_function_values):
                ax.annotate(f"{value:.02f}", 
                        (range(1,num_iters)[i], self._objective_function_values[i]),
                        ha='right',
                        rotation=-45)
            plt.ylim([self._objective_function_values[0]-5, self._objective_function_values[-1]+10])
            plt.xlim(0, num_iters)
        plt.title(title)
        plt.xlabel("# Iterations")
        plt.ylabel("Value of objective Function")
        plt.show()

    def solve_sudoku(self, A, y, c):
        print()
        w_star = A.T@y + c
        w_star_t = w_star.reshape((81,9)) #reshape for argmax
        self.sudoku.cells = 1 + np.argmax(w_star_t, axis=1).reshape((9,9))

    def plot_matrices(self):
        fig, axs = plt.subplots(3,2)
        axs[0, 0].spy(self.A_row)
        axs[0, 0].set_title('A_row')

        axs[0, 1].spy(self.A_col)
        axs[0, 1].set_title('A_col')

        axs[1, 0].spy(self.A_box)
        axs[1, 0].set_title('A_box')

        axs[1, 1].spy(self.A_color)
        axs[1, 1].set_title('A_color')

        axs[2, 0].spy(self.A_cell)
        axs[2, 0].set_title('A_cell')

        axs[2, 1].spy(self.A_clue)
        axs[2, 1].set_title('A_clue')

        fig.suptitle("Different parts of the A matrix")
        plt.show()

        plt.figure()
        plt.spy(self.A)
        plt.title("A Matrix")
        plt.show()

class Backtracking(object):
    def __init__(self, sudoku):
        self.sudoku = sudoku
        self.cells = self.sudoku.cells
        self.iterations = 0
        self.max_iterations = 1e10

    
    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, max_iterations):
        if max_iterations >= 81:
            self._max_iterations = max_iterations
        else:
            raise ValueError("max_iterations must be bigger or equal to 81 \
(min number of iterations to certanly solve the sudoku)")

    # check if sudoku is valid
    def check_board(self, grid):
        not_valid = False
        for y in range(len(grid)):
            for x in range(len(grid[y])):
                temp_grid = grid.copy() # copy for valid check -> we dont want to
                                        # wrong results if number in grid only if
                                        # number more than one time in grid
                if not grid[y][x] == 0 and not self.check_valid(y, x, grid[y][x]):
                    not_valid = True
                    print(f"Have a better look at cell {y+1}, {x+1} with the value {temp_grid[y][x]}")
                self.cells[y][x] = temp_grid[y][x]
        if not_valid:
            print("Does not look correct to me... ;(")
            return False
        return True

    def get_empty_cell(self):
        for y in range(len(self.cells)):
            for x in range(len(self.cells[y])):
                if self.cells[y][x] == 0:
                    return y,x
        return False

    def check_valid(self,y,x,number):
        #check row:
        self.cells[y][x] = 0 #no double count of number itself
        row = self.cells[y,:]
        if number in row:
            return False
        
        #ceck col:
        col = self.cells[:,x]
        if number in col:
            return False
        
        #cehck box:
        box = self.cells[(y//3)*3:(y//3)*3+3:1,(x//3)*3:(x//3)*3+3:1].flatten()
        if number in box:
            return False
                
        #check color:
        # color_cells = self.cells[y%3::3, x%3::3].flatten()
        # if number in color_cells:
        #     return False

        return True

    
    def solve(self):
        self.iterations += 1
        if self.iterations >= self.max_iterations:
            raise ValueError("Max number of iterations reached. Increase number of iterations!")
        # No print saves a lot of time ;)
        #print(f"Backtrack iterations: {self.iterations:06.02f}", end="\r") 
        cell = self.get_empty_cell()
        #print(cell, self.cells[cell[0]][cell[1]])
        if cell == False:
            return True
        
        y,x = cell
        for number in range(1,10):
            if self.check_valid(y,x,number):
                self.cells[y][x] = number
                if self.solve():
                    return True
                self.cells[y][x] = 0 
        return False