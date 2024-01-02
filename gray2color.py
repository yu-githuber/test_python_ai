import cv2
import os
from PIL import Image
import numpy as np
import numba
from numba import jit
import time
from tqdm import tqdm

'''
1.将单通道灰度图转位为3通道的png，读取灰度图里的值

'''

#彩图RGB配色盘 15类（限位器颜色已修改）原始颜色( 246, 220, 142 ),( 136,252,164 ), ( 142,242,244 )
COLOR = [(  0, 255,   0), (  0,   0, 142), (255, 150,   0), (250,  60,  60), (250,  60, 200),
         (200, 190,  30), (255, 255, 255), (169, 208, 142), (189, 215, 238), (189, 189, 238),
         (189, 150, 238), (  0,   0,   0),( 255,255,0 ),( 200, 56, 188 ), ( 64,222,240 )]

DRAW_COLOR = [(  0, 255,   0), (  0,   0, 142), (255, 150,   0), (250,  60,  60), (250,  60, 200),
            (200, 190,  30), (255, 255, 255), (169, 208, 142), (189, 215, 238), (189, 189, 238),
            (189, 150, 238), (0, 0, 0),( 211, 80, 80 ),( 200, 56, 188 ), ( 103,51,0 ),(0, 0, 0),(0,0,0)]  #嫣红，姹紫，棕色, 小灰灰

CITY_COLOR = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                                (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
                                (70, 130, 180), (220, 20, 60),(255, 0, 0),(0, 0, 142), (0, 0, 70),
                                (0, 60, 100),(0, 80, 100),(0, 0, 230),(119, 11, 32),(0, 0, 0)]  

#灰度图路径 png
gray_Dir = [  "/mnt/lustre/deeplearning/yzh/byd_dataset/RPL_Val_Images/mixgts/" ]

#3通道的mask图保存路径 png
save_mask_path = [ "/mnt/lustre/deeplearning/yzh/byd_dataset/RPL_Val_Images/mixgts_color_2/" ]

# @jit(nopython=True)
def convert_color(img_array,width,heigth):
    num=0
    for x in range(0,width):
        for y in range(0,heigth):
            rgb = img_array[x,y]
            r = rgb[ 0 ]
            g = rgb[ 1 ]
            b = rgb[ 2 ]
            #print(img_array[x,y])

            if r==0 :
                img_array[x,y] = DRAW_COLOR[0]#可行驶区域
            elif r==1 :
                img_array[x,y] = DRAW_COLOR[1]#汽车
            elif r==2 :
                img_array[x,y] = DRAW_COLOR[2]#柱子
            elif r==3 :
                img_array[x,y] = DRAW_COLOR[3]#路沿石
            elif r==4 :
                img_array[x,y] = DRAW_COLOR[4]#路边线
            elif r==5 :
                img_array[x,y] = DRAW_COLOR[5]#实线
            elif r==6:
                img_array[x,y] = DRAW_COLOR[6]#虚线
            elif r==7 :
                img_array[x,y] = DRAW_COLOR[7]#斑马线
            elif r==8 :
                img_array[x,y] = DRAW_COLOR[8]#导向箭头
            elif r==9:
                img_array[x,y] = DRAW_COLOR[9]#车位框线
            elif r==10 :
                img_array[x,y] = DRAW_COLOR[10]#减速带
            elif r==11:
                img_array[x,y] = DRAW_COLOR[11]#其他
            elif r==12:
                img_array[x,y] = DRAW_COLOR[12]#其他
            elif r==13:
                img_array[x,y] = DRAW_COLOR[13]#其他
            elif r==14:
                img_array[x,y] = DRAW_COLOR[14]#其他
            elif r==15:
                img_array[x,y] = DRAW_COLOR[15]#其他
                print("odd颜色配值：",r,f"x:{x}    y:{y}")
                num=num+1
            else:
                img_array[x,y] = DRAW_COLOR[16]#青色，其他类别
                print("其他颜色配值：",r,f"x:{x}    y:{y}")
                num=num+1

            # print(img_array[x,y])
    print(f"num:  {num}")

def cityscapes_color(img_array,width,heigth):
    for x in range(0,width):
        for y in range(0,heigth):
            rgb = img_array[x,y]
            r = rgb[ 0 ]
            g = rgb[ 1 ]
            b = rgb[ 2 ]
            #print(img_array[x,y])
            if r==0 :
                img_array[x,y] = CITY_COLOR[0]#可行驶区域
            elif r==1 :
                img_array[x,y] = CITY_COLOR[1]#汽车
            elif r==2 :
                img_array[x,y] = CITY_COLOR[2]#柱子
            elif r==3 :
                img_array[x,y] = CITY_COLOR[3]#路沿石
            elif r==4 :
                img_array[x,y] = CITY_COLOR[4]#路边线
            elif r==5 :
                img_array[x,y] = CITY_COLOR[5]#实线
            elif r==6:
                img_array[x,y] = CITY_COLOR[6]#虚线
            elif r==7 :
                img_array[x,y] = CITY_COLOR[7]#斑马线
            elif r==8 :
                img_array[x,y] = CITY_COLOR[8]#导向箭头
            elif r==9:
                img_array[x,y] = CITY_COLOR[9]#车位框线
            elif r==10 :
                img_array[x,y] = CITY_COLOR[10]#减速带
            elif r==11:
                img_array[x,y] = CITY_COLOR[11]#其他
            elif r==12:
                img_array[x,y] = CITY_COLOR[12]#其他
            elif r==13:
                img_array[x,y] = CITY_COLOR[13]#其他
            elif r==14:
                img_array[x,y] = CITY_COLOR[14]#其他
            elif r==15:
                img_array[x,y] = CITY_COLOR[15]#其他
            elif r==16:
                img_array[x,y] = CITY_COLOR[16]#其他
            elif r==17:
                img_array[x,y] = CITY_COLOR[17]#其他
            elif r==18:
                img_array[x,y] = CITY_COLOR[18]#其他
            else:
                img_array[x,y] = CITY_COLOR[19]#

            # print(img_array[x,y])
    

def gt_gray2color(gray_path,color_path):
    gray_list = os.listdir(gray_path)
    gray_list.sort()
    print("共有灰度图：", len(gray_list))

    with tqdm(total=len(gray_list), desc="上色中", leave=True, unit='img', unit_scale=True) as pbar:
        for i,name in enumerate(gray_list):
            img_name = name[:-4]
            png_path = os.path.join(gray_path, img_name + '.png')
            image_gray = Image.open(png_path).convert('RGB')
            width,heigth = image_gray.size
            mask_np = image_gray.load()
            # print(np.unique(seg))
            convert_color(mask_np,width,heigth)
            image_gray.save( os.path.join(color_path, img_name + '.png') )
            # cv2.imwrite(os.path.join(color_path, img_name + '.jpg'), new_array)
            pbar.update(1)


def find_gray2color(gray_path,color_path):
    gray_list = os.listdir(gray_path)
    gray_list.sort()
    print("共有灰度图：", len(gray_list))

    for i,name in enumerate(gray_list):
        img_name = name[:-4]
        png_path = os.path.join(gray_path, img_name + '.jpg')#png, jpg
        image_gray = Image.open(png_path).convert('RGB')
        array = np.array(image_gray)
        print(np.unique(array))

        width,heigth = image_gray.size
        mask_np = image_gray.load()
        cityscapes_color(mask_np,width,heigth)
        # convert_color(mask_np,width,heigth)
        image_gray.save( os.path.join(color_path, img_name + '.png') )

'''
if __name__ == '__main__':

    # 生成15分类的灰度图标签
    # gt_png2gray(mask_path,save_gray_path)
    for i in range(len(gray_Dir)):
        print("========第",  i+1,"  /  %d"%len(gray_Dir),"批========")

        if not os.path.exists(save_mask_path[i]):
            os.makedirs(save_mask_path[i])
        # 给灰度标签打底原图上色
        gt_gray2color(gray_Dir[i],save_mask_path[i])
        # find_gray2color(gray_Dir[i],save_mask_path[i])
'''

if __name__=='__main__':
        gt_gray='/home/yzh/work/semidrive/infer_outputnamed.jpg'
        save_path='/home/yzh/work/semidrive/'
        image_gray = Image.open(gt_gray).convert('RGB')
        width,heigth = image_gray.size
        print(f"width:{width}   height:{heigth}")
        mask_np = image_gray.load()
        print(f"mask_np type: {type(mask_np)}")
        # print(np.unique(seg))
        convert_color(mask_np,width,heigth)
        image_gray.save( os.path.join(save_path, 'infer_named_color.png') )
