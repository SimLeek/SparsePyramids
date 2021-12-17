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
    r = display(0, fps_limit=30)

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

            '''
            #cv basic corner detector. Not scale invariant.
            
            gray = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)

            corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
            corners = np.int0(corners)

            for i in corners:
                x, y = i.ravel()
                cv2.circle(img, (x, y), 3, 255, -1)

            r.update(img, 'cv2 corners')'''

            '''
            # cv orb corner detector. Should be scale invariant, but less robust.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            orb = cv2.ORB_create()
            kp = orb.detect(gray, None)

            img = cv2.drawKeypoints(gray, kp, None, color=(0,255,0), flags=0)

            r.update(img, 'cv2 orb')
            '''

            gimg = rgb_to_grayscale(timg)
            e_img = image_to_edge_pyramid(gimg, levels=2)
            chan_img = rchan.forward(e_img)
            dog_img = chan_img[:, 1:, ...] - chan_img[:, :-1, ...]
            max_dog = torch.max(dog_img, dim=1)
            min_dog = torch.min(dog_img, dim=1)
            interest_dog = max_dog.values[:, None, ...] + torch.abs(min_dog.values[:, None, ...])
            r.update(format_torch_image_for_display(interest_dog), 'interest_dog')
            blur_dog = hat_torch(interest_dog, 5)

            r.update(format_torch_image_for_display(blur_dog), 'blur_dog')
            interest_dog = interest_dog - blur_dog/1.5

            k = 1000
            inph = interest_dog.shape[2] - interest_dog.shape[2] % 6
            inpw = interest_dog.shape[3] - interest_dog.shape[3] % 6
            kernel = (inph//3, inpw//3)
            if maxpool1 is None:
                maxpool1 = torch.nn.MaxPool2d(kernel_size=kernel, stride=kernel, return_indices=True)
                unpool1 = torch.nn.MaxUnpool2d(kernel_size=kernel, stride=kernel)


            inp1 = interest_dog[:, :, :inph, :inpw]
            inp2 = interest_dog[:, :, inph//6:inph//6*5, inpw//6:inpw//6*5]

            out1, ind1 = maxpool1.forward(inp1)
            out2, ind2 = maxpool1.forward(inp2)

            #fixation
            if out1_old is not None:
                lesser = inp1.ravel()[ind1]<(inp1.ravel()[ind1_old]+img_var*2.0)
                better = inp1.ravel()[ind1]>(inp1.ravel()[ind1_old]+img_var*2.0)
                ind1[lesser] = ind1_old[lesser]
                out1[lesser] = out1[lesser]
            out1_old = out1
            ind1_old = ind1
            if out2_old is not None:
                lesser = inp2.ravel()[ind2] < (inp2.ravel()[ind2_old] + img_var * 2.0)
                better = inp2.ravel()[ind2] > (inp2.ravel()[ind2_old] + img_var * 2.0)
                ind2[lesser] = ind2_old[lesser]
                out2[lesser] = out2[lesser]
            out2_old = out2
            ind2_old = ind2

            # discard weak points
            out1[out1<img_var*2.0] = 0
            out2[out2<img_var*2.0] = 0

            # Note: discard lower values of out1 and out2, especially if they're still while others are moving.
            # Parts of the image with the same color will keep the same random fixation point, since they're not
            # exceeding the variance of the image noise.

            out1 = unpool1.forward(out1, ind1)
            out2 = unpool1.forward(out2, ind2)

            form_dog = format_torch_image_for_display((gimg[:, :, :inph, :inpw]+out1*255)/500.0)
            form_dog2 = format_torch_image_for_display((gimg[:, :, inph//6:inph//6*5, inpw//6:inpw//6*5]+out2*255)/500.0)

            r.update(form_dog, 'form_dog')
            r.update(form_dog2, 'form_dog2')

            t1 = time.time()
            print(f"fps:{1.0/(t1-t0)}")
            t0=t1