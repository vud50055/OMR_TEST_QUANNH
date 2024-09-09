import cv2
import numpy as np
import utlis

utlis.initializeTrackbars()
pathImage=r"C:\Users\Qs\PycharmProjects\ORM\.venv\TEST\Picture\3.jpg"
img=cv2.imread(pathImage)
img = cv2.resize(img, (920, 1301))
widthImg=img.shape[1]
heightImg=img.shape[0]
img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
thres = utlis.valTrackbars()# GET TRACK BAR VALUES FOR THRESHOLDS
imgThreshold = cv2.Canny(imgBlur, 100, 255)
#cv2.imshow("Threshhold",imgThreshold)# APPLY CANNY BLUR

# kernel = np.ones((5, 5))
# imgDial = cv2.dilate(imgThreshold, kernel, iterations=4)  # APPLY DILATION
# imgThreshold = cv2.erode(imgDial, kernel, iterations=4)  # APPLY EROSION
imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
imgContours=cv2.drawContours(imgContours,contours,-1,(0,255,0),4)# FIND ALL CONTOURS
fullcontour, RecContour=utlis.find_rectangular_contours(imgBigContour,contours)
phan1_contour,phan2_contour,phan3_contour=utlis.phanloaidapan(RecContour)

#imgBigContour=utlis.draw_contours_and_display_areas(imgBigContour,RecContour)
img_phan1=img.copy()
img_phan2=img.copy()
img_phan3=img.copy()
imgBigContour=utlis.draw_contours_and_display_areas(imgBigContour,RecContour)
cv2.imshow("big",imgBigContour)
img_phan1=cv2.drawContours(img_phan1,phan1_contour,-1,(0,255,0),4)
img_phan2=cv2.drawContours(img_phan2,phan2_contour,-1,(0,255,0),4)
img_phan3=cv2.drawContours(img_phan3,phan3_contour,-1,(0,255,0),4)

phan1_contour=utlis.loccontour(phan1_contour)
phan2_contour=utlis.loccontour(phan2_contour)
phan3_contour=utlis.loccontour(phan3_contour)

imgstack1=utlis.catanh(img_phan1,phan1_contour)
imgstack2=utlis.catanh(img_phan2,phan2_contour)
imgstack3=utlis.catanh3(img_phan3,phan3_contour)


cv2.imshow("1.1",imgstack1[0])
cv2.imshow("1.2",imgstack1[1])
cv2.imshow("1.3",imgstack1[2])
cv2.imshow("1.4",imgstack1[3])

cv2.imshow("2.1",imgstack2[0])
cv2.imshow("2.2",imgstack2[1])
cv2.imshow("2.3",imgstack2[2])
cv2.imshow("2.4",imgstack2[3])

cv2.imshow("3.1",imgstack3[0])
cv2.imshow("3.2",imgstack3[1])
cv2.imshow("3.3",imgstack3[2])
cv2.imshow("3.4",imgstack3[3])
cv2.imshow("3.5",imgstack3[4])
cv2.imshow("3.6",imgstack3[5])

# phan1=np.hstack((imgstack1[0],imgstack1[1],imgstack1[3]))
# phan2=np.hstack((imgstack2[0],imgstack2[1],imgstack2[3]))
# phan3.1=np.hstack((imgstack3[0],imgstack3[1],imgstack3[2]))
# phan3.2=np.hstack((imgstack3[3],imgstack3[4],imgstack3[5]))
#
# show1=vstack((phan1,phan2))
# show2=vstack((phan3.1,phan3.2))








cv2.waitKey(0)
