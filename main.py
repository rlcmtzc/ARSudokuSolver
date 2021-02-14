import cv2
from utils.processing_utils import *
from utils.sudoku import Sudoku
from utils.number_classifier import NumberClassifier
import time
import argparse


def detect_sudoku_on_image(image_path, font_scale_divisor = 3.5, deskew = False, show=False):
    train_image = cv2.imread("Data/train.png")
    train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
    classifier = NumberClassifier(train_image,deskew=deskew)
    preprcessor = PreProcessor(classifier, 600, 600)

    img = cv2.imread(image_path)
    start_time = time.time()

    grid = preprcessor.get_grid(img, show)
    print(grid)
    clues = grid.copy()
    sudoku = Sudoku(clues)
    sudoku.solve("backtrack", max_iterations=1e6)
    final_image = preprcessor.draw_grid(sudoku.cells - grid, img, x_offset=20, y_offset=15, font_scale_divisor=font_scale_divisor, show=show)

    end_time = time.time()
    print(f"Sudoku solved in {end_time - start_time:0.05} seconds!")
    preprcessor.show_image(final_image)



def live_detect_sudoku(font_scale_divisor = 3.5, deskew = False):
    cap = cv2.VideoCapture(0)
    train_image = cv2.imread("Data/train.png")
    train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
    classifier = NumberClassifier(train_image, deskew=deskew)
    preprcessor = PreProcessor(classifier, 600, 600)

    correct_sudoku = np.zeros((9,9))
    prev_grid = np.zeros((9,9))
    
    sudoku = Sudoku(np.zeros((9,9))) # Initial Sudoku
    while(True):
        _, frame = cap.read()
        frame = cv2.resize(frame, None, None, fx=0.6, fy=0.6)
        grid = preprcessor.get_grid(frame)
        
        if grid is not None:
            if not np.all(prev_grid == grid): #only solve when the grid changed
                clues = grid.copy()
                sudoku = Sudoku(clues)
                try:
                    sudoku.solve("backtrack", max_iterations=5000)
                except ValueError: #If not solved fast enough show just the frame
                    frame = cv2.resize(frame, None, None, fx=1.5, fy=1.5)
                    cv2.imshow('Sudoku Solver',frame)
                    continue
            else:
                prev_grid = grid.copy()
            if np.sum(sudoku.cells) == 405:
                correct_sudoku = sudoku.cells
                frame = preprcessor.draw_grid(sudoku.cells - grid, frame, x_offset=20, y_offset=15, font_scale_divisor=font_scale_divisor)
            else:
                frame = preprcessor.draw_grid(correct_sudoku - grid, frame, x_offset=20, y_offset=15, font_scale_divisor=font_scale_divisor)

        frame = cv2.resize(frame, None, None, fx=1.5, fy=1.5)
        cv2.imshow('Sudoku Solver',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



parser = argparse.ArgumentParser(description='Input to main file')
parser.add_argument('mode', metavar='image_mode', type=int, default=0,
                    help='if mode is 1 image Mode is selected. if mode is 0 video mode is selected. Default 0 ')
parser.add_argument('--path', dest='image_path', type=str, default="",
                    help='If mode is 1 enter path to the image')
parser.add_argument('--fsd', dest='font_scale_divisor', type=float, default=3.5,
                    help='Changes the fontsize of the added numbers')
parser.add_argument('--deskew', dest='deskew', type=bool, default=False,
                    help='deskews the numbers when detecting. Helps with handwritten numbers or a high skew of the image')
parser.add_argument('--debug', dest='debug', type=bool, default=False,
                    help='Shows all the debug steps inbetween final result works only in image mode')

args = parser.parse_args()

if args.mode == 1:
    if args.image_path == "":
        raise ValueError("Image Mode is selected please enter the path")
    
    detect_sudoku_on_image(args.image_path, font_scale_divisor=args.font_scale_divisor, deskew=args.deskew, show=args.debug)
elif args.mode == 0:
    live_detect_sudoku(font_scale_divisor=args.font_scale_divisor, deskew=args.deskew)
else:
    raise ValueError(f"mode {args.mode} is not availabel please use 0 (video) or 1 (image)")

