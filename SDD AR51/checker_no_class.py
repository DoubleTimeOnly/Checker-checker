import numpy
import cv2
import imutils
import timeit

def wrapper(func,*args,**kwargs):
    def wrapped():
        return func(*args,**kwargs)
    return wrapped


def Distance(p1, p2):
    return numpy.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def ColorMask(image, lowerBoundary, upperBoundary):
    lower = numpy.array(lowerBoundary, dtype='uint8')
    upper = numpy.array(upperBoundary, dtype='uint8')
    mask = cv2.inRange(image, lower, upper)
    return mask


def DetectCircles(orig,dp=1,mindist=20,cannyThreshold=150, accumulator=15, minRadius=0, maxRadius=100):
        gray = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, mindist,
                                   param1=cannyThreshold, param2=accumulator,
                                   minRadius=minRadius, maxRadius=maxRadius)

        if circles is not None:
            # print 'circles found',len(circles)
            circles = numpy.uint16(numpy.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(orig, center, 1, (0, 100, 100), 1)
                # circle outline
                radius = i[2]
                print radius
                # radius = 0
                cv2.circle(orig, center, radius, (255, 0, 255), 1)
                cv2.imshow('circle',orig)

def DetectColoredCircles(orig,colors,dp=1,mindist=20,cannyThreshold=150, accumulator=15, minRadius=0, maxRadius=100):
    for a,color in enumerate(colors):
        frameDelta = ColorMask(orig,color[0],color[1])
        kern_len = 10
        kernel = numpy.ones((kern_len,kern_len), numpy.uint8)
        frameDelta = cv2.morphologyEx(frameDelta, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('frameDelta{0}'.format(a),frameDelta)

        masked = cv2.bitwise_and(orig,orig,mask=frameDelta)

        gray = cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY)
        # for i in range(4):
        #     gray = cv2.bilateralFilter(gray,11,17,17)
        # cv2.imshow('warp_gray',gray)

        # gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

        # cv2.imshow('warp_thresh',gray)
        # edge = cv2.Canny(gray,75,150)

        # cv2.imshow('warp_canny',edge)
        circles = cv2.HoughCircles(frameDelta, cv2.HOUGH_GRADIENT, dp, mindist,
                                   param1=cannyThreshold, param2=accumulator,
                                   minRadius=minRadius, maxRadius=maxRadius)

        if circles is not None:
            print 'circles found',len(circles)
            circles = numpy.uint16(numpy.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(orig, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                print radius
                # radius = 0
                cv2.circle(orig, center, radius, (255, 0, 255), 3)
                cv2.imshow('circle',orig)


def GetTransformMatrix(pts,ratio):
    rect = numpy.zeros((4, 2), dtype='float32')  # clockwise starting from top-left

    # top-left and bottom-right
    s = pts.sum(axis=1)
    # print s
    rect[0] = pts[numpy.argmin(s)]
    rect[2] = pts[numpy.argmax(s)]

    diff = numpy.diff(pts, axis=1)
    # print diff
    rect[1] = pts[numpy.argmin(diff)]
    rect[3] = pts[numpy.argmax(diff)]
    # print rect
    rect *= ratio

    (tl, tr, br, bl) = rect
    topWidth = Distance(tl, tr)
    botWidth = Distance(bl, br)
    maxWidth = int(max(topWidth, botWidth))
    leftHeight = Distance(tl, bl)
    rightHeight = Distance(tr, br)
    maxHeight = int(max(leftHeight, rightHeight))

    # Make transform array
    dst = numpy.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    transformMatrix = cv2.getPerspectiveTransform(rect, dst)
    return transformMatrix,maxWidth,maxHeight

# Grayscale -> bilateral filter -> closing
def Preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    for i in range(0):
        gray = cv2.bilateralFilter(gray, 11, 17, 17)  # noise removal via bilateral filter
    cv2.imwrite('./debug/bilateral.png', gray)

    # for i in range(5,20,2):
    #     temp = gray.copy()
    #     kernel = numpy.ones((i, i), numpy.uint8)
    #     temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
    #     cv2.imshow('closing{0}'.format(i),temp)

    # kernel = numpy.ones((3, 3), numpy.uint8)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('closing', gray)
    return gray


def FindLines(image1,image2,edges):
    lines = cv2.HoughLines(edges, 1, numpy.pi / 180, 200)
    lines = sorted(lines, key=lambda x: x[0][0],reverse=True)
    vert=[]
    hori=[]
    prev_vert = 0
    prev_hori = 0
    for line in lines:
        # print line
        rho, theta = line[0]
        a = numpy.cos(theta)
        b = numpy.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # angle taken from vertical
        if -numpy.pi/8 < theta%numpy.pi < numpy.pi/8:
            if abs(rho - prev_vert) > 20:
                cv2.line(image1, (x1, y1), (x2, y2), 255, 1)
                vert.append(line)
                prev_vert = rho

        elif 3*numpy.pi/8 < theta%numpy.pi < 5*numpy.pi/8:
            if abs(rho - prev_hori) > 20:
                cv2.line(image2, (x1, y1), (x2, y2), 255, 1)
                hori.append(line)
                prev_hori = rho

    #     vert = sorted(vert, key=lambda x: x[0][0], reverse=True)
    #     hori = sorted(hori, key=lambda x: x[0][0], reverse=True)
    # for line in vert:
    #     print line
    # for line in hori:
    #     print line

def PerspectiveTransform(orig,screenCnt,ratio=1):
    # Perspective transform
    pts = screenCnt.reshape(4, 2)
    # cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
    # cv2.imwrite('./debug/outline.png', image)
    # cv2.imshow('image', image)


    # Warp
    # print 'pts and ratio'
    # print pts
    # print ratio
    transformMatrix, maxWidth, maxHeight = GetTransformMatrix(pts, ratio)

    warp = cv2.warpPerspective(orig, transformMatrix, (maxWidth, maxHeight))
    cv2.imwrite('./debug/out.png', warp)
    return warp

def main(filePath):
    try:
        image = cv2.imread(filePath)
    except AttributeError:
        print 'Invalid filepath, using default'
        return

    orig = image.copy()

    ratio = image.shape[0] / 300.  # remember the pixel ratio of old:new

    image = imutils.resize(image, height=300)  # resize image for quicker analysis
    # image = cv2.UMat(image)


    gray = Preprocessing(image)
    edged = cv2.Canny(gray, 30, 100)  # Canny edge detection
    cv2.imwrite('./debug/edge.png', edged)
    cv2.imshow('canny', edged)

    try:
        (c1, contours, hierarchy) = cv2.findContours(edged, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)
    # opencv 2.x legacy function call
    except ValueError:
        (contours, _) = cv2.findContours(edged, cv2.RETR_TREE,
                                         cv2.CHAIN_APPROX_SIMPLE)
    ##contour sorting
    temp = []
    numContours = 12
    if len(contours) < 10:
        numContours = len(contours)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:numContours]
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    for c in contours:
        moments = cv2.moments(c)
        cx = int(moments['m10'] / moments['m00'])
        temp.append((cv2.contourArea(c), c))
        # temp.append((cv2.contourArea(c) - (10 * (cx - width)), c))
    screenCnt = 1

    # sort contours by their new weight
    temp = sorted(temp, key=lambda x: x[0], reverse=True)[:5]

    # Check for four vertices
    for c in temp:
        c = c[1]
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * perimeter, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    try:
        warp = PerspectiveTransform(orig,screenCnt,ratio=ratio)
        # print type(screenCnt)

    except AttributeError:
        cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
        cv2.imshow('exit_contours', image)
        print '[error]: no valid contours'
        print 'Exiting'
        cv2.waitKey()
        return


    xm=8
    ym=8
    width=orig.shape[1]
    height = orig.shape[0]
    warp_orig = orig.copy()
    for i in range(1, ym + 2):
        cv2.line(warp_orig, (0, i * height / (ym + 2)), (width, i * height / (ym + 2)), (255, 0, 0), 1)
    for i in range(1, xm + 2):
        cv2.line(warp_orig, (i * width / (xm + 2), 0), (i * width / (xm + 2), height), (255, 0, 0), 1)

    gray = Preprocessing(warp)
    edges = cv2.Canny(gray,30,150)
    w,h,c = warp.shape
    # print 'shape',w,h
    hmask = numpy.zeros((w,h), numpy.uint8)
    vmask = numpy.zeros((w,h), numpy.uint8)
    # extract horizontal and vertical lines and draw them on vmask,hmask
    FindLines(vmask,hmask,edges)
    # cv2.imshow('hmask',hmask)
    # cv2.imshow('vmask', vmask)
    mask = cv2.bitwise_and(hmask,vmask)
    # print hmask.shape
    # print vmask.shape
    hv_side = numpy.hstack((hmask,vmask))
    grid = cv2.bitwise_or(hmask,vmask)
    gi_side = numpy.hstack((grid,mask))
    points = cv2.findNonZero(mask)
    # cv2.imshow('bit mask',mask)
    # print 'points',len(points),points.shape
    # print points
    # cv2.imwrite('./debug/grid_intersection.png',gi_side)
    cv2.imwrite('./debug/horizontal_vertical_masks.png', hv_side)
    # cv2.imshow('binary_mask',mask)

    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    warp = cv2.bitwise_or(warp,mask)
    warp_mask = numpy.hstack((mask,warp))
    # cv2.imwrite('./debug/intersection_overlay.png',warp)
    cv2.imshow('mask', warp_mask)

    old_height = -1000
    points_sorted = []
    for i in range(len(points)):
        point = points[i][0]
        # print i,point
        if point[1] - old_height > 20:
            points_sorted.append([point])
            old_height = point[1]
        else:
            points_sorted[len(points_sorted)-1].append(point)
    warp_orig = warp.copy()
    for row in range(len(points_sorted)-1):
        points_sorted[row] = sorted(points_sorted[row],key=lambda x:x[0])

    for r,row in enumerate(range(len(points_sorted)-1)):
        for c,column in enumerate(range(len(points_sorted[row])-1)):
            square = [points_sorted[row][column],points_sorted[row+1][column],points_sorted[row+1][column+1],points_sorted[row][column+1]]
            # print 'square'
            # print square
            # square = numpy.array(square)
            # print type(warp)
            # box = PerspectiveTransform(warp_orig,square)
            p1 = square[0]
            p2 = square[2]
            print r,p1,c,p2
            box = (warp[square[0][1]:square[2][1], square[0][0]:square[2][0]]).copy()
            red = [(42,42,150),(110,110,255)]
            white = [(200,200,200),(255,255,255)]
            colors = [red,white]
            h,w = box.shape[:2]
            # print 'width:',w
            gray_box = cv2.cvtColor(box,cv2.COLOR_BGR2GRAY)
            edge = cv2.Canny(gray_box,50, 100)
            DetectCircles(box, dp=1, mindist=w, cannyThreshold=100, accumulator=10, minRadius=w/3, maxRadius=w/2)

            square_edge = numpy.hstack((box,cv2.cvtColor(edge,cv2.COLOR_GRAY2BGR)))

            cv2.imshow('box',imutils.resize(square_edge,height=300))
            cv2.waitKey()


# cv2.imshow('warp',warp)

    cv2.waitKey(0)

    # threshold('./debug/out.png', './debug/text.png')
    # text = pytesseract.image_to_string(Image.open('./debug/text.png'))




# main-----------------------------------------------------------------------------

if __name__ == '__main__':
    file1 = './TestImages/chessboard4.jpg'
    file2 = './TestImages/ex1.jpeg'
    file3 = './TestImages/chessboard3.jpg'
    # main('chessboard4.jpg')
    wrapped = wrapper(main,file2)
    num = 1
    print timeit.timeit(wrapped,number=num)/num

    ''' 
    find outer contour
    maybe check contour hierarchy to see if there is one big contour inside or many small contours 
    color/circle detection for checker piece
    check for box around checker piece, check for parent contour if possible
    determine grid size in pixels from that
    determine approximate board size from that
    '''