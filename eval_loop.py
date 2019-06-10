import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
from PIL import Image

parser = argparse.ArgumentParser(description="PyTorch VDSR Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--dataset", default="Set5", type=str, help="dataset name, Default: Set5")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

scales = [2,3,4]

image_list = glob.glob(opt.dataset+"input/*.jpg")
image_list.sort(reverse=False)
gt_list = glob.glob(opt.dataset+"GT/*.jpg")
gt_list.sort(reverse=False)
# print(image_list)

avg_psnr_predicted = 0.0
avg_psnr_bicubic = 0.0
avg_elapsed_time = 0.0
count = 0.

img_size = (1280,720)

for i in range(len(image_list)):

    image_name = image_list[i]
    gt_name = gt_list[i]

    count += 1.

    print("Processing ", image_name)
    # im_gt_y = sio.loadmat(image_name)['im_gt_y']
    # im_b_y = sio.loadmat(image_name)['im_b_y']

    im_b_y = np.zeros((15,img_size[1],img_size[0]), dtype=np.float32)

    for j in range(5):
        if(i-2+j>=0) and (i-2+j<len(image_list)):
            img = Image.open(image_list[i-2+j])
            im_b_y[3*j:3*(j+1),:,:] = np.moveaxis(np.array(img.resize(img_size)), 2, 0)
    gt_img = Image.open(gt_name)
    im_gt_y = np.moveaxis(np.array(gt_img.resize(img_size)), 2, 0)

    im_gt_y = im_gt_y.astype(float)
    im_b_y = im_b_y.astype(float)

    img_test = im_b_y[6:9,:,:]
    psnr_bicubic = PSNR(img_test, im_gt_y,shave_border=0)
    avg_psnr_bicubic += psnr_bicubic

    im_input = im_b_y/255.

    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[1], im_input.shape[2])

    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    HR = model(im_input)
    elapsed_time = time.time() - start_time
    avg_elapsed_time += elapsed_time

    HR = HR.cpu()

    im_h_y = HR.data[0].numpy().astype(np.float32)

    im_h_y = im_h_y * 255.
    im_h_y[im_h_y < 0] = 0
    im_h_y[im_h_y > 255.] = 255.
    # im_h_y = im_h_y[0,:,:]

    psnr_predicted = PSNR(img_test, im_h_y,shave_border=0)
    avg_psnr_predicted += psnr_predicted

    result = Image.fromarray((np.moveaxis(im_h_y, 0, 2)).astype(np.uint8))
    result.save('result/{:04d}.bmp'.format(i))

print("Dataset=", opt.dataset)
print("PSNR_predicted=", avg_psnr_predicted/count)
print("PSNR_bicubic=", avg_psnr_bicubic/count)
print("It takes average {}s for processing".format(avg_elapsed_time/count))
