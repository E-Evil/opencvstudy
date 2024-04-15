import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
def cv_show(name,img):
#name是窗口名
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return
putin=cv.imread("f1.jpg")
print(putin.shape)
img=cv.medianBlur(putin,3)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret,binary=cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
cv_show("bin",binary)
binary=255-binary
cv_show("ant_bin",binary)
kernel=np.ones((9,9),dtype=np.uint8)
bin_clo=cv.morphologyEx(binary,cv.MORPH_CLOSE,kernel,3)
cv_show("bin_clo",bin_clo)
num,labels,stats,centroids=cv.connectedComponentsWithStats(bin_clo,connectivity=8)
bin_clo=cv.cvtColor(bin_clo,cv.COLOR_GRAY2BGR)
for i ,stat in enumerate(stats):
    cv.rectangle(bin_clo,(stat[0],stat[1]),(stat[0]+stat[2],stat[1]+stat[3]),(0,0,255),3)
    cv.putText(bin_clo,str(i+1),(stat[0],stat[1]+25),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    print(i+1,stats[i][-1])
check=np.hstack((img,bin_clo))
cv_show("check",check)
output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
for i in range(1, num):
    if stats[i][-1]>=stats[11][-1] and stats[i][-1]<=stats[4][-1]:
        mask = labels == i
        output[:, :, 0][mask] =0
        output[:, :, 1][mask] =255
        output[:, :, 2][mask] =255
cv_show("s",output)
result=cv.add(img,output)
cv_show("result",result)
