import os
import cv2
import skimage.io as io
import numpy as np
from lxml import etree


'''
CVAT XML 파일에서 라벨링 이미지를 만드는 소스코드
'''

def parse_anno_file_to_image(cvat_xml, *image_name, num=0):
    image_names=[]
    image_ids=[]
    image_shapes=[]
    points_datas=[]
    maskLabel=[]
    poly_num=[]
    
    root = etree.parse(cvat_xml).getroot()
    
    for image_tag in root.findall("image"):
        number = 0
        if not image_tag.find("polygon") == None:
            image_names.append(image_tag.attrib["name"])
            image_ids.append(image_tag.attrib["id"])
            image_shapes.append([int(image_tag.attrib["height"]), int(image_tag.attrib["width"])])
            for poly in image_tag.findall("polygon"):
                number+=1
                points_datas.append(poly.attrib["points"])
            poly_num.append(number)       
    print(poly_num)
    for points in points_datas:
        vectors=[]
        for point in points.split(";"):
            x, y = point.split(",")
            vectors.append(np.array((float(x), float(y))).astype(int))
        maskLabel.append(vectors)
        
    
    for index, shape in enumerate(image_shapes):
        height, width = shape
        background = np.zeros((height, width), dtype="uint8")
        
        
        if poly_num[index] == 1:
            mask= maskLabel[index]
            mask= np.expand_dims(mask, axis=0)
            result = cv2.drawContours(background, [mask], -1, color=(128, 128, 128), thickness=3)
            result = cv2.fillPoly(result, [mask], color=(255, 255, 255))
        else:
            for i in range(poly_num[index]):
                mask= maskLabel[index+i]
                mask= np.expand_dims(mask, axis=0)
                if i ==0:
                    result = cv2.drawContours(background, [mask], -1, color=(128, 128, 128), thickness=3)
                else:
                    result = cv2.drawContours(result, [mask], -1, color=(128, 128, 128), thickness=3)
                result = cv2.fillPoly(result, [mask], color=(255, 255, 255))
                
                
                
        image = io.imread(os.path.join(os.getcwd(), "redvelvet", image_names[index])) # annotations == Originals annotations2 == BTS annotations3 ==redvelvet
        
        if True:
            numbering = num+index
        else:
            numbering = index
            
        io.imsave(os.path.join(os.getcwd(), "x", f"{numbering}"+".jpeg"), image)
        io.imsave(os.path.join(os.getcwd(), "y", f"{numbering}"+".jpeg"), result)
        if not poly_num[index]==1:
            index+=1
            
        
    

    

if __name__ == "__main__":
    num = len(os.listdir(os.path.join(os.getcwd(), "x")))
    parse_anno_file_to_image(os.path.join(os.getcwd(), "xmlFiles", "annotations3.xml"), num=num)