#imports-----------------------------------------------------------------------------
import pytesseract
import cv2
from PIL import Image


#functions-----------------------------------------------------------------------------
def threshold(sourcePath,destinationPath):
    filePath = sourcePath

    image = cv2.imread(filePath)
    grayscale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # Noise removal via gaussian filter
    for i in range(0,2):
        grayscale = cv2.bilateralFilter(grayscale, 11, 17, 17)

    # Apply adaptive thresholding
    th3 = cv2.adaptiveThreshold(grayscale,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)    
    cv2.imwrite(destinationPath,th3)


#main-----------------------------------------------------------------------------

if __name__ == '__main__':
    threshold('./debug/out.png','./debug/th3.png')
    print '[FINAL TEXT]'
    print pytesseract.image_to_string(Image.open('./debug/th3.png'))
    print '[END TEXT]'
    
