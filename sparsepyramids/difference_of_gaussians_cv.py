import numpy as np

from edge_pyramid import image_to_edge_pyramid
import torch
from recursive_pyramidalize import RecursiveChanDepyramidalize2D

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


import cv2
import numpy as np


def DoG(img):
    # run a 5x5 gaussian blur then a 3x3 gaussian blr
    blur5 = cv2.GaussianBlur(img, (5, 5), 0)
    blur3 = cv2.GaussianBlur(img, (3, 3), 0)

    DoGim = blur5 - blur3
    return DoGim


def compareImages(in1, in2):
    output1 = in2 * -1 * in1
    return output1

if __name__ == '__main__':
    from sparsepyramids.tests.videos import test_video
    from displayarray import display
    from sparsepyramids.edge_pyramid import hat_torch
    import time
    r = display(0, fps_limit=30, size=(9999,9999))

    rchan = RecursiveChanDepyramidalize2D()
    from rand_std_sparse import RandStdSparse
    from guarantee_max_sparse import RandMaxSparse

    class sparsifier(torch.nn.Module):
        def __init__(self):
            super(sparsifier, self).__init__()
            self.sp = RandStdSparse(
                avg_percent_activation=0.0001, make_sparse_tensor=False, mean=0.0, rand_portion=0.0
            )
            #self.sp2 = RandMaxSparse(out_percent_nonzero=0.001, make_sparse_tensor=False)

        def forward(self, x):
            x = self.sp(x)
            #x = self.sp2(x)
            # test out_percent_nonzero: torch.count_nonzero(x)/x.numel()
            # should always be less than out_percent_nonzero
            return x


    sp = sparsifier()

    t1=t0 = time.time()
    for u in r:
        if u:
            img = next(iter(u.values()))[0]
            timg = numpy_to_torch(img)
            gimg = rgb_to_grayscale(timg)
            e_img = image_to_edge_pyramid(gimg, levels=5)
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
            kdog = interest_dog.view(-1).topk(k).indices.view(-1, 1)
            indices = torch.cat((torch.zeros((k, 1), dtype=torch.int), torch.zeros((k, 1), dtype=torch.int), kdog // timg.shape[2], kdog % timg.shape[3]), dim=1)
            descriptors = torch.permute(dog_img, (0,2,3,1)).view(-1, dog_img.shape[1])[kdog]


            out = torch.zeros_like(interest_dog)
            out.view(-1)[kdog] = 1
            #kdog = kdog.values
            #kdog = (kdog - torch.min(kdog)) / (torch.max(kdog) - torch.min(kdog) + 1e-7)

            form_dog = format_torch_image_for_display(out)
            r.update(form_dog, 'form_dog')
            t1 = time.time()
            print(f"fps:{1.0/(t1-t0)}")
            t0=t1