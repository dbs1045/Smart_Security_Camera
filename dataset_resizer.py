import cv2
import os

def maximum_file(resizeF= True):
    if resizeF:
        files = os.listdir(os.path.join(os.getcwd(), "x_resize")) # dataset 개수
    else:
        files = os.listdir(os.path.join(os.getcwd(), "x"))
    filenames=[]
    for file in files:
        if file.split(".")[1] == "jpeg":
            filenames.append(int(file.split(".")[0]))
    num = max(filenames)+1
    return num


def x_data_generator(num):
    for i in range(num):
        try:
            image = cv2.imread(os.path.join(os.getcwd(), "x", f"{i}.jpeg"))
            image = cv2.resize(image, (256, 256))
            cv2.imwrite(os.path.join(os.getcwd(), "x_resize", f"{i}.jpeg"), image)
        except:
            continue
        
        
def y_data_generator(num):
    for i in range(num):
        try:
            image = cv2.imread(os.path.join(os.getcwd(), "y", f"{i}.jpeg"))
            image = cv2.resize(image, (256,256))
            cv2.imwrite(os.path.join(os.getcwd(), "y_resize", f"{i}.jpeg"), image)
        except:
            continue
        
if __name__=="__main__":
    num = maximam_file(False)
    x_data_generator(num)
    y_data_generator(num)