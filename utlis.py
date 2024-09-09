import cv2
import numpy as np



## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray, scale, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, lables[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (255, 0, 255), 2)
    return ver


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 100000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def drawRectangle(img, biggest, thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

    return img


def nothing(x):
    pass


def initializeTrackbars(intialTracbarVals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 225, 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 109, 255, nothing)



def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")

    return Threshold1,Threshold2


def find_rectangular_contours(image, contours):
    rect_contours = []  # Danh sách lưu các contour hình chữ nhật

    for contour in contours:
        if cv2.contourArea(contour) >20000 and cv2.contourArea(contour) < 300000:
            # Tính chu vi của contour
            peri = cv2.arcLength(contour, True)
            # Xấp xỉ hình dạng của contour
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Kiểm tra xem contour có 4 cạnh và diện tích lớn hơn một giá trị ngưỡng không
            if len(approx) == 4 or len(approx)==5:  # Hình chữ nhật sẽ có 4 đỉnh
                rect_contours.append(contour)

    # Vẽ các contour hình chữ nhật lên ảnh
    img_rectangles = image.copy()
    cv2.drawContours(img_rectangles, rect_contours, -1, (0, 255, 0), 4)

    return img_rectangles, rect_contours


def draw_contours_and_display_areas(image, contours):
    # Tạo bản sao của hình ảnh để vẽ contours
    output_image = image.copy()

    for contour in contours:
        # Tính diện tích của contour chính
        contour_area = cv2.contourArea(contour)

        # Vẽ contour chính trên hình ảnh
        cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)

        # Hiển thị diện tích của contour chính
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(output_image, f'Area: {contour_area:.2f}', (x, y+100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Tìm các contour con bên trong contour chính


    return output_image

def phanloaidapan(contours):
    phan1_contours = []
    phan2_contours =[]
    phan3_contours =[]
    for contour in contours:
        if cv2.contourArea(contour) >44000 and cv2.contourArea(contour) < 45000:
            phan1_contours.append(contour)
        elif cv2.contourArea(contour) >25000 and cv2.contourArea(contour) < 27000:
            phan2_contours.append(contour)
        elif cv2.contourArea(contour) >200000:
            phan3_contours.append(contour)
    return phan1_contours, phan2_contours, phan3_contours

def loccontour(contours):
    filtered_contours=[]

    threshold=200
    for i in range(len(contours)):
        # Lấy bounding box cho contour hiện tại
        x1, y1, w1, h1 = cv2.boundingRect(contours[i])

        # Kiểm tra xem contour hiện tại có gần tương đương với contour nào khác không
        similar = False
        for j in range(i + 1, len(contours)):
            # Lấy bounding box cho contour tiếp theo
            x2, y2, w2, h2 = cv2.boundingRect(contours[j])

            # So sánh vị trí và kích thước giữa hai bounding boxes
            if abs(x1 - x2) < threshold and abs(y1 - y2) < threshold and abs(w1 - w2) < threshold and abs(
                    h1 - h2) < threshold:
                similar = True
                break
        if not similar:
            filtered_contours.append(contours[i])
        for i in range(0,len(filtered_contours)):
            x1, y1, w1, h1 = cv2.boundingRect(filtered_contours[i])
            for j in range(i+1, len(filtered_contours)):
                x2, y2, w2, h2 = cv2.boundingRect(filtered_contours[j])
                if x1 > x2:
                    temp=filtered_contours[i]
                    filtered_contours[i]=filtered_contours[j]
                    filtered_contours[j] = temp

    return filtered_contours


def catanh(img,contours):
    crop_image=[]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        crop_image.append(img[y:y+h, x:x+w])
    return crop_image
def catanh3(img,contour):
    crop_image=[]
    x, y, w, h = cv2.boundingRect(contour[0])
    img = img[y:y + h, x:x + w]
    newWidth = img.shape[1] // 6
    for i in range(0,6):
        croped_image=img[:,newWidth*i:newWidth*(i+1)]
        crop_image.append(croped_image)
    return crop_image