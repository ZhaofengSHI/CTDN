import cv2
import os
import numpy as np
from PIL import Image

# COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
#                 [0, 0, 128]]

COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128]]           

# COLORMAP = [[0, 0, 0], [128, 0, 0]]     

# CLASSES = ['background', 'my left hand', 'my right hand', 'your left hand', 'your right hand']

# CLASSES = ['background', 'left hand', 'right hand', 'left hand object', 'right hand object', 'both hands object']

# CLASSES = ['background', 'hands']

CLASSES = ['background', 'left hand', 'right hand', 'hand object']



root_dir = './output_final_0.1/visor_hos'

mask_dir = 'pseudo_masks_visor_hos'
colormask_dir = 'pseudo_masks_visor_hos_color'

if not os.path.exists(os.path.join(root_dir,colormask_dir)):
    os.mkdir(os.path.join(root_dir,colormask_dir))

for i in range(len(os.listdir(os.path.join(root_dir,mask_dir)))):
    file_name = os.listdir(os.path.join(root_dir,mask_dir))[i]
    mask_gray = np.array(Image.open(os.path.join(root_dir,mask_dir,file_name)).convert("P"))
    print(mask_gray.shape)
    
    print(mask_gray.min(),mask_gray.max())

    mask_color= np.zeros((mask_gray.shape[0],mask_gray.shape[1],3))


    #进行染色
    for c in range(len(CLASSES)):

        mask_color[:, :, 2] += ((mask_gray[:, :] == c) * (COLORMAP[c][0])).astype('uint8')
        mask_color[:, :, 1] += ((mask_gray[:, :] == c) * (COLORMAP[c][1])).astype('uint8')
        mask_color[:, :, 0] += ((mask_gray[:, :] == c) * (COLORMAP[c][2])).astype('uint8')

    mask_color_name = file_name
    # mask_color = cv2.cvtColor(mask_color,cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(root_dir,colormask_dir,mask_color_name),mask_color)
    cv2.waitKey(250)
# cv2.imwrite(str(f"{folder}pred_{idx}.png"),pred_img)