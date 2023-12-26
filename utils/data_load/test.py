import cv2
import numpy as np
from tqdm import tqdm
import os
from multiprocessing import  Process

'''
conda activate last_mul_raters
cd Ours/utils/data_load/
python test.py

'''
def make_seg(original, seg, dest_Disc, dest_Cup, dest_original=None):
    img1 = cv2.imread(original, 0)
    if not os.path.exists(seg):
        seg = seg.replace('.jpg', '.tif')
    img2 = cv2.imread(seg, 0)

    if img1 is None:
        print('=========== img1 ===========')
        print(original)
        raise ValueError
    
    if img2 is None:
        print('=========== img2 ===========')
        print(seg)
        raise ValueError
    

    img3 = abs(img2-img1)


    if not os.path.exists(dest_original):
        img_ori = cv2.imread(original)
        img_ori = cv2.resize(img_ori, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(dest_original, img_ori)

        
    H, W = img3.shape
    img3 = np.reshape(img3, (H*W))
    img3 = np.where(img3 == 0, img3, 255)
    img3 = np.reshape(img3, (H,W))

    first_circle = []
    for h in range(H):
        tmp = []
        for w in range(W):
            if img3[h][w] != 0:
                if tmp == []:
                    tmp.append(w)
                    first_circle.append((h,w))
                else:
                    for each in tmp:
                        if w-1 in tmp:
                            tmp.append(w)
                            first_circle.append((h,w))
                            break

    for h in range(H):
        tmp = []
        for w in range(W-1,0,-1):
            if img3[h][w] != 0:
                if tmp == []:
                    tmp.append(w)
                    first_circle.append((h,w))
                else:
                    for each in tmp:
                        if w+1 in tmp:
                            tmp.append(w)
                            first_circle.append((h,w))
                            break
    
    img_disc1 = img3.copy()
    img_disc2 = img3.copy()
    img_cup1 = img3.copy()
    img_cup2 = img3.copy()
    for each in first_circle:
        img_cup1[each[0], each[1]] = 0
        img_cup2[each[0], each[1]] = 0
        


    # 制作对应的disc mask
    for h in range(H):
        flag = [0,0]
        for w in range(W):
            if flag == [1,1]:
                break
            if flag[0] == 0 and img_disc1[h][w] != 0:
                img_disc1[h][w:] = np.ones_like(img_disc1[h][w:]) * 100
                flag[0] = 1
            if flag[1] == 0 and img_disc2[h][W-w-1] != 0:
                # print('?')
                img_disc2[h][:W-w] = np.ones_like(img_disc2[h][:W-w]) * 100
                flag[1] = 1

    disc = img_disc1 + img_disc2
    empty = np.zeros_like(disc)
    disc = np.where(disc != 200, empty, 255)


    # 制作对应cup mask
    for h in range(H):
        flag = [0,0]
        for w in range(W):
            if flag == [1,1]:
                break
            if flag[0] == 0 and img_cup1[h][w] != 0:
                img_cup1[h][w:] = np.ones_like(img_cup1[h][w:]) * 100
                flag[0] = 1
            if flag[1] == 0 and img_cup2[h][W-w-1] != 0:
                img_cup2[h][:W-w] = np.ones_like(img_cup2[h][:W-w]) * 100
                flag[1] = 1

    cup = img_cup1 + img_cup2
    empty = np.zeros_like(cup)
    cup = np.where(cup != 200, empty, 255)




    disc = cv2.resize(disc, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    cup = cv2.resize(cup, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)


    
    cv2.imwrite(dest_Disc, disc)
    cv2.imwrite(dest_Cup, cup)


if __name__ == '__main__':
    root = '/data2/chengyi/dataset/Multiple_Annotation/RIGA/BinRushedcorrected/BinRushed/BinRushed4'
    seg_root = '/data2/chengyi/dataset/Multiple_Annotation/RIGA/BinRushedcorrected/small/set4/mask'
    img_root = '/data2/chengyi/dataset/Multiple_Annotation/RIGA/BinRushedcorrected/small/set4/input'
    data = {}
    # seg = [[] for _ in range()]
    for each in os.listdir(root):
        if 'image' not in each:
            continue
        if 'prime' in each:
            num = each.split('prime')[0][5:]
        else:
            num = each.split('-')[0][5:]
        if num not in data:
            data[num] = [each]
        else:
            data[num].append(each)
    
    



    # def sub(i):
    #     seg = os.path.join(root, 'image{}-{}.tif'.format(key, i))
    #     dest_cup = os.path.join(seg_root, 'image{}-{}-cup.jpg'.format(key, i))
    #     dest_disc = os.path.join(seg_root, 'image{}-{}-disc.jpg'.format(key, i))
    #     dest_original = os.path.join(img_root, 'image{}.jpg'.format(key, i))
    #     make_seg(original, seg, dest_Cup=dest_cup, dest_Disc=dest_disc, dest_original=dest_original)

    for key in tqdm(data):
        original = os.path.join(root, 'image{}prime.jpg'.format(key))
        # print(original)
        # raise ValueError

        process_list = []
        for i in range(6): 
            # image_name = 'Image' if i == 1 else 'image'
            p = Process(target=make_seg,args=(original, 
                                              os.path.join(root, 'image{}-{}.jpg'.format(key, i+1)), 
                                              os.path.join(seg_root, 'image{}-{}-cup.jpg'.format(key, i+1)), 
                                              os.path.join(seg_root, 'image{}-{}-disc.jpg'.format(key, i+1)), 
                                              os.path.join(img_root, 'image{}.jpg'.format(key, i+1)))) 
            p.start()
            process_list.append(p)

        for i in process_list:
            p.join()


        


    
    
