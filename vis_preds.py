import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
from PIL import Image

main_dir = sys.argv[1]
use_web = int(sys.argv[2])
#plt.figure(1); plt.figure(2); # Img overlay Gt, Pred
#plt.ion()
#plt.show()

def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 4)

def create_overlay(img, mask, colors):
    im= Image.fromarray(np.uint8(img))
    im= im.convert('RGBA')

    mask_color= np.zeros((mask.shape[0], mask.shape[1],3))
    if len(colors)==3:
        mask_color[mask==colors[1],0]=255
        mask_color[mask==colors[1],1]=255
        mask_color[mask==colors[2],0]=255
    else:
        mask_color[mask==colors[1],2]=255

    overlay= Image.fromarray(np.uint8(mask_color))
    overlay= overlay.convert('RGBA')

    im= Image.blend(im, overlay, 0.7)
    blended_arr= PIL2array(im)[:,:,:3]
    img2= img.copy()
    img2[mask==colors[1],:]= blended_arr[mask==colors[1],:]
    return img2[:, :, ::-1]

if not os.path.exists(main_dir+'overlay_sprt'):
    os.mkdir(main_dir+'overlay_sprt')
if not os.path.exists(main_dir+'overlay_qry'):
    os.mkdir(main_dir+'overlay_qry')

for f in sorted(os.listdir(main_dir+'sprt/')):
    print(f)
    pred = cv2.imread(main_dir+'qry_pred/'+f, 0)
    img = cv2.imread(main_dir+'qry/'+f)
    sprt_img = cv2.imread(main_dir+'sprt/'+f)
    sprt_gt = cv2.imread(main_dir+'sprt_lbl/'+f, 0)

    pred[pred!=1]=0
    pred[pred==1]=255
    pred = cv2.resize(pred, img.shape[:2], interpolation=cv2.INTER_NEAREST)
    overlay_qry_pred= create_overlay(img, pred, [0,255])
    
    if use_web == 1:
        overlay_sprt = sprt_img[:,:,::-1]
    else:
        sprt_gt[sprt_gt!=1]=0
        sprt_gt[sprt_gt==1]=255
        sprt_gt = cv2.resize(sprt_gt, sprt_img.shape[:2], interpolation=cv2.INTER_NEAREST)
        overlay_sprt= create_overlay(sprt_img, sprt_gt, [0,255])

    cv2.imwrite(main_dir+'overlay_sprt/'+f, overlay_sprt)
    cv2.imwrite(main_dir+'overlay_qry/'+f, overlay_qry_pred)
#    plt.figure(1); plt.imshow(overlay_qry_pred);
#    plt.figure(2); plt.imshow(overlay_sprt);

#    plt.draw()
#    plt.waitforbuttonpress(0)

