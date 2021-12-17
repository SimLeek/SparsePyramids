import math

import numpy as np

from edge_pyramid import image_to_edge_pyramid
import torch
from recursive_pyramidalize import RecursiveChanDepyramidalize2D
import cv2

def format_torch_image_for_display(t: torch.Tensor):
    # todo: move to displayarray
    assert len(list(t.shape)) == 3 or t.shape[
        0] == 1, "tensor to be displayed should be an image or have a batch size of one. If you want to display an entire batch, you can run a for loop over the batch dimension."
    t = torch.squeeze(t)
    if len(list(t.shape))==3:
        t = torch.permute(t, (1, 2, 0))
    tc = t.cpu().detach().numpy()
    return tc


def numpy_to_torch(np_img):
    torch_image = torch.FloatTensor(np_img)
    torch_image = torch.permute(torch_image, (2, 0, 1))[None,]

    return torch_image

def rgb_to_grayscale(t: torch.Tensor):
    assert t.shape[1] == 3

    # assuming bgr from opencv
    b = t[:, 0:1, ...]
    g = t[:, 1:2, ...]
    r = t[:, 2:3, ...]

    o = 0.299 * r + 0.587 * g + 0.114 * b
    return o

if __name__ == '__main__':
    from sparsepyramids.tests.videos import test_video
    from displayarray import display
    from sparsepyramids.edge_pyramid import hat_torch
    import time
    import itertools
    r = display(0, fps_limit=60)

    rchan = RecursiveChanDepyramidalize2D()
    maxpool1 = unpool1 = None

    t1=t0 = time.time()
    out1_old =out2_old =None
    ind1_old =ind2_old =None
    f1= f2=0
    for u in r:
        if u:
            img = next(iter(u.values()))[0]
            img_filtered = cv2.medianBlur(img, 5)  # improves edges in image better than most other de-noising algorithms

            timg = numpy_to_torch(img_filtered)
            timg_orig = numpy_to_torch(img)

            img_var = torch.sum(torch.abs(timg_orig - timg))/timg_orig.numel()  # variance of image needed for fixation

            # cv orb corner detector. Should be scale invariant, but less robust.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Todo: use lots of features and add fixation by using distance from previous feature * img variance
            #orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
            orb = cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE)
            kp = orb.detect(gray, None)

            img1 = cv2.drawKeypoints(gray, kp, None, color=(0, 255, 0), flags=0)
            r.update(img1, 'cv2 orb 0')

            qdict1 = dict()
            for k in kp:
                y, x = k.pt
                yq = int(y / (img.shape[0] // 3))
                xq = int(x / (img.shape[1] // 3))
                if xq < 3 and yq < 3:
                    if yq in qdict1.keys():
                        if xq in qdict1[yq].keys():
                            if k.response > qdict1[yq][xq].response:
                                qdict1[yq][xq] = k
                        else:
                            qdict1[yq][xq] = k
                    else:
                        qdict1[yq] = dict()
                        qdict1[yq][xq] = k

            qdict2 = dict()
            for k in kp:
                y, x = k.pt
                yq = int(math.floor((y-(img.shape[0] // 6)) / (img.shape[0] // 3)))
                xq = int(math.floor((x-(img.shape[1] // 6)) / (img.shape[1] // 3)))
                if 0 <= xq < 2 and 0 <= yq < 2:
                    if yq in qdict2.keys():
                        if xq in qdict2[yq].keys():
                            if k.response > qdict2[yq][xq].response:
                                qdict2[yq][xq] = k
                        else:
                            qdict2[yq][xq] = k
                    else:
                        qdict2[yq] = dict()
                        qdict2[yq][xq] = k

            kp1 = list(itertools.chain(*[list(x.values()) for x in qdict1.values()]))
            img1 = cv2.drawKeypoints(gray, kp1, None, color=(0,255,0), flags=0)
            r.update(img1, 'cv2 orb 1')

            kp2 = list(itertools.chain(*[list(x.values()) for x in qdict2.values()]))
            img2 = cv2.drawKeypoints(gray, kp2, None, color=(0, 255, 0), flags=0)
            r.update(img2, 'cv2 orb 2')

            t1 = time.time()
            print(f"fps:{1.0/(t1-t0)}")
            t0=t1