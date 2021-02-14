import cv2
import numpy as np
import os

class NumberClassifier():
    def __init__(self, train_image, deskew=False):
        self._image_size = 28
        self.deskew_image = deskew

        if not os.path.isfile("Data/model.xml"):
            self._train_cells = []
            self._train_labels = []
            self._train_data = self.get_train_data(train_image)
            print(np.array(self._train_data).shape)
            self.train()
        else:
            #print("ModelFound")
            self._svm = cv2.ml.SVM_load("Data/model.xml")
            #print("ModelLoaded")
        pass
    
    @property
    def image_size(self):
        return self._image_size

    @image_size.setter
    def image_size(self, image_size):
        if not isinstance(image_size, int):
            raise TypeError("NumberClassifier: image_size must be of type int")
        if image_size <= 0:
            raise ValueError("NumberClassifier: image_size must be bigger than 0")
        self._image_size = image_size

    def deskew(self, img):
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            # no deskewing needed. 
            return img.copy()
        # Calculate skew based on central momemts. 
        skew = m['mu11']/m['mu02']
        # Calculate affine transform to correct skewness. 
        M = np.float32([[1, skew, -0.5*self._image_size*skew], [0, 1, 0]])
        # Apply affine transform
        img = cv2.warpAffine(img, M, (self._image_size, self._image_size), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return img

    def get_train_data(self, train_image):
        height, width = train_image.shape
        
        current_label = 0
        hog = []
        for y in range(0,height, self._image_size):
            current_label += 1
            for x in range(0,width, self._image_size):
                current_cell = train_image[y:y+self._image_size, x:x+self._image_size]
                current_cell = self.deskew(current_cell)

                if np.sum(current_cell) != 0:
                    #h = self.get_HOG(current_cell)
                    hog.append(self.get_HOG(current_cell))
                    self._train_cells.append(current_cell)
                    self._train_labels.append(current_label)
        return hog

    def get_HOG(self, img):
        #width, height, _ = img.shape
        winSize = (20, 20)
        blockSize = (10,10)
        blockStride = (5,5)
        cellSize = (10,10)
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        useSignedGradients = True

        hog = cv2.HOGDescriptor(winSize,
                                blockSize,
                                blockStride,
                                cellSize,
                                nbins,
                                derivAperture,
                                winSigma,
                                histogramNormType,
                                L2HysThreshold,
                                gammaCorrection,
                                nlevels, 
                                useSignedGradients)
        return hog.compute(img)


    def classify(self, img):
        print(img)

    def train(self):
        print("Start Training")
        # Set up SVM for OpenCV 3
        self._svm = cv2.ml.SVM_create()
        # Set SVM type
        self._svm.setType(cv2.ml.SVM_C_SVC)
        # Set SVM Kernel to Radial Basis Function (RBF) 
        self._svm.setKernel(cv2.ml.SVM_RBF)
        # Set parameter C
        C = 100
        gamma = 0.9
        self._svm.setC(C)
        # Set parameter Gamma
        self._svm.setGamma(gamma)

        # Train SVM on training data  
        self._svm.train(np.array(self._train_data), cv2.ml.ROW_SAMPLE, np.array(self._train_labels))
        self._svm.save("Data/model.xml")

    def predict(self, img):
        
        img = cv2.copyMakeBorder( img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, 0)
        img = cv2.resize(img, (self._image_size, self._image_size))
        if self.deskew_image:
            img = self.deskew(img)

        #cv2.imshow("number", img)
        #if not cv2.waitKey():
        #    exit()
        
        img_hog = self.get_HOG(img)
        pred = self._svm.predict(np.array([img_hog]))
        #print(pred)
        pred_number = pred[1].flatten()[0]
        return pred_number