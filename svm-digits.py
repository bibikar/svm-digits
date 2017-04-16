from sklearn import datasets,svm
import cv2
import numpy as np

digits = datasets.load_digits()
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data, digits.target)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
	print("capture open failed")
	exit()
while (1):
	
	_, fram = cap.read()
	fram_s = fram.shape
	fram_x_offset = fram_s[1]-fram_s[0]
	fram = fram[:,fram_x_offset:fram_x_offset+fram_s[1]]

	grey = cv2.cvtColor(fram,cv2.COLOR_BGR2GRAY);
	_,grey = cv2.threshold(grey,127,255,cv2.THRESH_TOZERO);
	#grey = cv2.GaussianBlur(grey,(41,41),11)
	#grey = cv2.adaptiveThreshold(grey,256,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,2)

	subsize = (300, 300)
	grey_first = (int(grey.shape[0]/2) - int(subsize[0]/2),int(grey.shape[1]/2) - int(subsize[1]/2))
	eight = cv2.resize(grey[grey_first[0]:grey_first[0]+subsize[0],grey_first[1]:grey_first[1]+subsize[1]],(8, 8), interpolation=cv2.INTER_NEAREST);
	eight_show = cv2.resize(eight,(512,512));
	
	prediction = clf.predict(eight.reshape(-1,1))
	print(prediction)
	cv2.putText(fram,str(prediction)[1],(100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,5,255),2,cv2.LINE_AA)
	cv2.rectangle(grey,(grey_first[1],grey_first[0]),(grey_first[1]+subsize[1],grey_first[0]+subsize[0]),(0,0,0))
	cv2.imshow('fram',grey)
	cv2.imshow('filter',eight_show)
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break


