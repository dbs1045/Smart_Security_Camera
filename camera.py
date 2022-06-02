import cv2
import numpy as np
import os
from Unet import Unet
import skimage.io as io
from skimage import transform as trans
import pygame as pg

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def alarm(image, Threshold=0.4):
    matrix = image>0.6
    _, array2 = np.unique(matrix, return_counts=True)
    try:
        if array2[1]/(array2[1]+array2[0]) >Threshold:
            print("사람일 확률: ",array2[1]/(array2[1]+array2[0]))
            f = "siren.mp3" 
            filename = os.path.join(os.getcwd(), 'music', f)
            pg.mixer.init()
            pg.mixer.music.load(filename=filename)
            pg.mixer.music.play(1)
            return array2[1]/(array2[1]+array2[0])
    except:
        pass
    
def callBack(camera_mode):
    return not camera_mode

'''카메라 구동함수'''
def video_capture(model, camera_mode=False):
    background = io.imread(os.path.join(os.getcwd(), "Background", "mountain.jpeg"))
    background = trans.resize(background, (256, 256))
    background = (255*background).astype("uint8")
    cv2.namedWindow("INPUT ESC", cv2.WINDOW_NORMAL)
    cv2.moveWindow("INPUT ESC", 300, 15)
    cv2.createTrackbar("Threshold", "INPUT ESC", 0,100, lambda x: x)
    cv2.setTrackbarPos("Threshold", "INPUT ESC", 40)
    try:
        print("카메라를 구동합니다.")
        cap=cv2.VideoCapture(0)
        cap.set(480,640)
    except:
        print("카메라 구동실패.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print('비디오 읽기 오류')
            break
        frame = cv2.resize(frame, (256,256))
        frame = cv2.flip(frame,1)
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame2 = np.expand_dims(frame2.astype(float)/255, axis=0)
        test = model.predict(frame2)
        
        rate = alarm(test, float(cv2.getTrackbarPos("Threshold", "INPUT ESC")/100))
        try:
            rate = round(rate, 3)
        except:
            rate ="error"
        test = (test[0,:,:,0]*255).astype("uint8")
        
        if camera_mode:
            hsvframe = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h,s,v =cv2.split(hsvframe)
            seg = cv2.bitwise_and(v, test)
            seg = cv2.merge([h,s,seg])
            seg = cv2.cvtColor(seg, cv2.COLOR_HSV2BGR)
            
            # hsvframe2 = cv2.cvtColor(background, cv2.COLOR_RGB2HSV)
            # h,s,v =cv2.split(hsvframe2)
            # seg2 = cv2.bitwise_and(v, 255-test)
            # seg2 = cv2.merge([h,s,seg2])
            # seg2 = cv2.cvtColor(seg2, cv2.COLOR_HSV2BGR)
            result = seg
        else:
            result = test
            
        cv2.putText(result, f"People Rate: {rate}",(40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1, cv2.LINE_AA)    
        result = cv2.resize(result, (480,640))
        frame = cv2.resize(frame, (480,640))
        if not camera_mode:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        result = cv2.hconcat([result, frame])
        cv2.imshow("INPUT ESC", result)
        
        k = cv2.waitKey(50) & 0x75
        if k == 17:
            break
        elif k==32:
            camera_mode = callBack(camera_mode)
            
        
    cap.release()
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(os.getcwd(), "cameraSeg", "seg.jpeg"), frame)


if __name__== "__main__":
    seg = video_capture(Unet(True), camera_mode=False)
    