import cv2
import numpy

def main():

    image = cv2.imread('./TestImages/connect2.jpg')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    for i in range(2):
        gray = cv2.bilateralFilter(gray,11,17,17)
    edge = cv2.Canny(gray,30,200)
    cv2.imshow('gray',gray)
    cv2.imshow('edge',edge)
    rows = gray.shape[0]
    '''
        gray: Input image (grayscale).
        circles: A vector that stores sets of 3 values: xc,yc,r for each detected circle.
        HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV.
        dp = 1: The inverse ratio of resolution.
        min_dist = gray.rows/16: Minimum distance between detected centers.
        param_1 = 200: Upper threshold for the internal Canny edge detector.
        param_2 = 100*: Threshold for center detection.
        min_radius = 0: Minimum radius to be detected. If unknown, put zero as default.
        max_radius = 0: Maximum radius to be detected. If unknown, put zero as default.
    '''
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=100, param2=30,
                               minRadius=20, maxRadius=100)

    if circles is not None:
        circles = numpy.uint16(numpy.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            print radius
            radius = 0
            cv2.circle(image, center, radius, (255, 0, 255), 3)

    cv2.imshow("detected circles", image)
    cv2.waitKey(0)



if __name__=='__main__':
    main()