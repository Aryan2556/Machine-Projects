from lib import *

mpHands=mp.solutions.hands 
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils   
ptime,ctime=0,0
x1,y1,x2,y2,b1,b2=0,0,0,0,0,0
volrange=volume.GetVolumeRange()
minvol=volrange[0]
maxvol=volrange[1]
vol,volBar=0,150
cap=cv2.VideoCapture(0)
cap.set(3,640) 
cap.set(4,760)
cap.set(10,200)
while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results1=hands.process(imgRGB)


    if results1.multi_hand_landmarks:
        for hand in results1.multi_hand_landmarks:
            for id,ln in enumerate(hand.landmark):
                finger=id
                h,w,c=img.shape
                cx,cy=int(ln.x*w),int(ln.y*h)

                if finger==8:
                    x1=cx
                    y1=cy
                    cv2.circle(img,(x1,y1),12,(0,255,0),cv2.FILLED)
                elif finger==4:
                    x2=cx
                    y2=cy
                    cv2.circle(img,(x2,y2),12,(0,0,255),cv2.FILLED)
                else:
                    pass
                cv2.line(img,(x1,y1),(x2,y2),(255,133,233),4)
            fx,fy=(x1+x2)//2,(y1+y2)//2
            cv2.circle(img,(fx,fy),9,(255,255,0),cv2.FILLED)
            mpDraw.draw_landmarks(img,hand,mpHands.HAND_CONNECTIONS)
    
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    length=math.hypot(x2-x1,y2-y1)
    vol=np.interp(length,[30 ,200],[minvol,maxvol])
    vol1=np.interp(length,[50 ,300],[400,150])
    volper=np.interp(length,[50 ,300],[0,100])

    volume.SetMasterVolumeLevel(vol,None)

    cv2.putText(img,f'Fps:{int(fps)}',(10,100),cv2.FONT_ITALIC,2,(255,0,0),3) #BGR
    cv2.putText(img,f'Volume:{int(volper)} %',(10,190),cv2.FONT_ITALIC,2,(185,180,0),3) #BGR
    
    cv2.imshow("Tracking Output",img)
    if cv2.waitKey(20) & 0xFF==ord("f"):
        break

cap.release()
cv2.destroyAllWindows()


