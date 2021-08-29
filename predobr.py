import cv2
import os
from PIL import Image
size = 120, 120
for i in os.listdir(os.getcwd()+'\\data_ident'):
    for j in os.listdir(os.getcwd()+ '\\data_ident\\' + i):
        for k in os.listdir(os.getcwd() + f'\\data_ident\\{i}\\{j}'):
            im = Image.open(os.getcwd()+ f'\\data_ident\\{i}\\{j}\\{k}')
            im.thumbnail(size, Image.ANTIALIAS)
            width, height = im.size
            m = -0.5
            xshift = abs(m) * width
            new_width = width + int(round(xshift))
            im = im.transform((new_width, height), Image.AFFINE,
                                (1, m, -xshift if m > 0 else 0, 0, 1, 0), Image.BICUBIC)
            im.show()
            break
            im.save(os.getcwd() + f'\\data_ident\\{i}\\{j}\\{k}')
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            im.save(os.getcwd()+ f'\\data_ident\\{i}\\{j}\\new_{k}')
            print(k)
        break
    break