import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import time

# helper class for displaying the Sudoku and converting
class Sudoku(object):
    def __init__(self, clues):
        # define the initial values
        self.cells = np.zeros((9,9))
        self.cells[clues[:,0]-1,clues[:,1]-1] = clues[:,2]
        
        # for plotting
        self.clues = np.zeros((9,9))
        self.given_clues = clues
        self.clues[clues[:,0]-1,clues[:,1]-1] = clues[:,2]

        self.fig = None
        self.ax = None
        self.title = None

    def to_x(self):
        ''' transforms the cell digits into the flattened volumetric representation
            return x\in\R^{9^3}
        '''
        # initialize with zeros
        x = np.zeros((9, 9, 9))

        # generate meshgrid to assign value for every position
        xx, yy = np.meshgrid(np.arange(9), np.arange(9))

        # -1 because sudoku starts at 1 and vol starts at 0
        self.cells = self.cells.astype(int)
        x[yy, xx, self.cells - 1] = 1
        x[self.cells==0, :] = 0
        return x.ravel()

    def from_x(self, x):
        ''' sets the cells according to the maximal entry of the distribution along the inner dimension
            param x\in\R^{9^3}
        '''
        # reshape to volumetric representation and take the argmax
        self.cells = x.reshape(9,9,9).argmax(axis=-1) + 1

    def show(self, title):
        ''' plots the current state of the Killer Sudoku
        '''
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax.clear()
            
        # setup the sudoku layout
        major_ticks = np.arange(0,10,3)
        minor_ticks = np.arange(0,10,1)
        self.ax.set_xticks(major_ticks)
        self.ax.set_xticks(minor_ticks, minor=True)
        self.ax.set_yticks(major_ticks)
        self.ax.set_yticks(minor_ticks, minor=True)
        self.ax.grid(True, 'major', linewidth=3, color='k')
        self.ax.grid(True, 'minor', linewidth=1, color='k')
        self.ax.get_xaxis().set_ticklabels([])
        self.ax.get_yaxis().set_ticklabels([])
        self.ax.set_aspect('equal', 'box')

        # plot the numbers
        cmap = plt.get_cmap('Set3', 9)
        for i in range(9):
            for j in range(9):
                self.ax.axvspan(j, j+1, ymin=(9-i-1)/9, ymax=(9-i)/9, alpha=1, color=cmap(j%3 + 3*(i%3)))
                if self.clues[i,j]:
                    self.ax.text(j+0.5,8.5-i, f'{self.cells[i,j].astype(np.int8):d}', 
                        va='center', ha='center',fontsize=12, fontweight='bold', 
                        color='black' if self.cells[i,j]==self.clues[i,j] else 'red')
                elif self.cells[i,j] > 0:
                    self.ax.text(j+0.5,8.5-i, f'{self.cells[i,j].astype(np.int8):d}', 
                        va='center', ha='center',fontsize=12, fontstyle='italic', 
                        color='blue')

        self.ax.set_xlim(0, 9)
        self.ax.set_ylim(0, 9)
        if self.title is not None:
            self.title += f" {title}"
            self.ax.set_title(self.title)
        plt.show()
        self.fig.canvas.draw()


    def solve(self, method="DA", plot_matrices=False, max_iterations=1e10, iterations=5):
        bt_solver = Backtracking(self) # for later Checking if the sudoku is correct
        if method == "DA":
            da_solver = DualAffineSolver(self)
            da_solver.iterations = iterations
            da_solver.solve()
            print()
            if plot_matrices:
                da_solver.plot_matrices()
                da_solver.plot_objective_function_values()
                
            self.title = "Dual Affine Scaling"
        elif method == "backtrack":
            #bt_solver = Backtracking(self)
            bt_solver.max_iterations = max_iterations
            bt_solver.solve()
            print()
            self.title = "Backtracking"
        else:
            raise ValueError(f"Solving method {method} is not implemented.\
Following methods are available:\nDA: Dual Affine Scaling\nbacktrack: Backtracking")
        if bt_solver.check_board(self.cells):
            print("Looks good to me ;)")

        

# Dual Affine Scaling Solver Class
class DualAffineSolver(object):
    def __init__(self, sudoku):
        self.sudoku = sudoku
        self.clues = self.sudoku.given_clues
        self.iterations = 5
        self.objective_function_values = []
        

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
        clues_onehot = np.zeros((self.clues[:,2].size, 9))
        clues_onehot[np.arange(self.clues[:,2].size),self.clues[:,2] - 1] = 1

        clues_onehot_index = np.asarray([self.clues[:,0] - 1, self.clues[:,1] - 1]).T

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
            self.objective_function_values.append(objective_function_value)
            if objective_function_value < pref_objective_function_value + self.epsilon:
                break
            pref_objective_function_value = objective_function_value
        return y

    def plot_objective_function_values(self):
        num_iters = len(self.objective_function_values) + 1
        title = f"Values of objective Function for {num_iters - 1} iterations"
        fig, ax = plt.subplots()
        plt.grid()
        ax.plot(range(1,num_iters), self.objective_function_values, color="springgreen")

        if num_iters <= 12: # Add Annotations only if there are a few iterations
            ax.scatter(range(1,num_iters), self.objective_function_values, color="cyan")
            for i, value in enumerate(self.objective_function_values):
                ax.annotate(f"{value:.02f}", 
                        (range(1,num_iters)[i], self.objective_function_values[i]),
                        ha='right',
                        rotation=-45)
            plt.ylim([self.objective_function_values[0]-5, self.objective_function_values[-1]+10])
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
        color_cells = self.cells[y%3::3, x%3::3].flatten()
        if number in color_cells:
            return False

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
    


if __name__ == '__main__':
    # load the clues for the killer sudoku 
    # the clues are saved as a list with entries:
    #   clues[i][0] y position of clue
    #   clues[i][1] x position of clue
    #   clues[i][2] digit of clue
    # have a look at the plotting function to see more details
    
    import time
    # easy
    clues = np.load('./clues_easy.npy')
    # create the Killer Sudoku
    s = Sudoku(clues)
    start_time = time.time()
    s.solve("DA", iterations=20, plot_matrices=False)
    #s.solve("backtrack", max_iterations=1e6)
    end_time = time.time()
    print(f"Easy Sudoku solved in {end_time - start_time:0.05} seconds!")
    # plot the Killer Sudoku
    plt.ion();s.show(title="Easy Clues");plt.ioff();plt.show()

    ### BONUS TASK ###
    # hard
    clues = np.load('./clues_hard.npy')
    # create the Killer Sudoku
    s = Sudoku(clues)
    start_time = time.time()
    #s.solve("DA", iterations=20, plot_matrices=False)
    s.solve("backtrack", max_iterations=1e6)
    end_time = time.time()
    print(f"Hard Sudoku solved in {end_time - start_time:0.05} seconds!")
    # plot the Killer Sudoku
    plt.ion();s.show(title="Hard Clues");plt.ioff();plt.show()