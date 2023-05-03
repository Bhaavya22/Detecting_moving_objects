import cv2, pandas
import time
from datetime import datetime

first_frame=None
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["Start","End"])

video=cv2.VideoCapture(0)
a=0
while True:
    a=a+1
    check,frame=video.read()
    #print(check)
    #print(frame)
    status=0

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)  #to smooth it becoz that removes noise and inc accuracy in the calculations of the diff


    if first_frame is None:
        first_frame=gray
        continue  #continue to the begining of the loop

    delta_frame=cv2.absdiff(first_frame,gray)

    thresh_frame=cv2.threshold(delta_frame,30,255, cv2.THRESH_BINARY)[1]

    thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)

    (cnts,_)= cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour)<10000:
            continue
        status=1
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    status_list.append(status)
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())
    cv2.imshow("capturing",gray)
    cv2.imshow("delta_frame",delta_frame)
    cv2.imshow("_threshdelta",thresh_frame)
    cv2.imshow("colorframe",frame)
    

    key=cv2.waitKey(1)
    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break

    #print(status)

print(status_list)
print(times)

for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("Times.csv")
#time.sleep(5)
video.release()
cv2.destroyAllWindows()