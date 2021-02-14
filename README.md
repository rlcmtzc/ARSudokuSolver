# ARSudokuSolver
Augmented Reality Sudokusolver with OpenCV 

## Setup:
1) Install Python >= 3.8
2) Install requirements
´´´ 
$pip install -r requirements.txt 
´´´

## Run script for images:
Run the main script:
```
$python main.py 1 --path=<path_to_image>
```
For help run:
```
$python main.py -h
```

You can change the number size with the fsd (font scale divisor) as follows:
```
$python main.py 1 --path=<path_to_image> --fsd=3.0 
```

If not all numbers get detected correctly you can try to deskew them while detecting as follows:
```
$python main.py 1 --path=<path_to_image> --fsd=3.0 --deskew=True
```

You can look at the debug images as follows:
```
$python main.py 1 --path=<path_to_image> --fsd=3.0 --deskew=True --debug=True
```

## Run script on webcam:
Run the main script:
```
$python main.py 0
```

For help run:
```
$python main.py -h
```

You can change the number size with the fsd (font scale divisor) as follows:
```
$python main.py 1 --path=<path_to_image> --fsd=3.0 
```

If not all numbers get detected correctly you can try to deskew them while detecting as follows:
```
$python main.py 1 --path=<path_to_image> --fsd=3.0 --deskew=True
```
 
