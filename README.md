# Spatio-Temporal Transformer Network for Video Restoration
This is implementation of the paper [Spatio-Temporal Transformer Networkfor Video Restoration](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwiOq8qu-eHiAhXIlIsKHbdGCHwQFjAAegQIAhAC&url=http%3A%2F%2Fopenaccess.thecvf.com%2Fcontent_ECCV_2018%2Fpapers%2FTae_Hyun_Kim_Spatio-temporal_Transformer_Network_ECCV_2018_paper.pdf&usg=AOvVaw0lhDjBkIQbpuxCIE3k9a0Q)
## Dependencies
The code was developed on python3 with pytorch and pillow libraries. Please visit [installation guide](https://pytorch.org/get-started/locally/) for pytorch installation. For installing the pillow simple type `pip3 install pillow` on terminal

## Dataset
The code was trained on [Deep Video Deblurring](https://arxiv.org/pdf/1611.08387)'s
dataset which can be accessed from this [link](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset.zip). Unzip it into a desired
folder. Alternatively, you can place your own videos under <br/>
`dataset/qualitative_datasets/[video_file_name]/input` as input and <br/>
`dataset/qualitative_datasets/[video_file_name]/GT` as ground truth videos <br/>
as frame extracted videos. This dataset structure can be used for both training and testing. You can extract a video into frames using ffmpeg with
the following command <br/>
`ffmpeg -i file.mpg -r 1/1 $foldername/%04d.jpg` <br/>
where `$foldername` is desired folder for frame extraction

## Training
For training you need to call main_spatio.py file with the corresponding option parameters. <br/>
usage: main_spatio.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS]
                      [--lr LR] [--step STEP] [--cuda] [--resume RESUME]
                      [--start-epoch START_EPOCH] [--threads THREADS]
                      [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
                      [--pretrained PRETRAINED] [--gpus GPUS]
                      [--dataset DATASET] <br/>

optional arguments: <br/>

  --batchSize BATCHSIZE Training batch size <br/>
  --nEpochs NEPOCHS     Number of epochs to train for <br/>
  --lr LR               Learning Rate. Default=0.1 <br/>
  --step STEP           Sets the learning rate to the initial LR decayed by <br/>
                        momentum every n epochs, Default: n=10 <br/>
  --cuda                Use cuda? <br/>
  --resume RESUME       Path to checkpoint (default: none) <br/>
  --start-epoch START_EPOCH
                        Manual epoch number (useful on restarts) <br/>
  --threads THREADS     Number of threads for data loader to use, Default: 1 <br/>
  --momentum MOMENTUM   Momentum, Default: 0.9 <br/>
  --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        Weight decay, Default: 1e-4 <br/>
  --pretrained PRETRAINED
                        path to pretrained model (default: none) <br/>
  --gpus GPUS           gpu ids (default: 0) <br/>
  --dataset DATASET     the folder where dataset can be found with specified <br/>
  --model MODEL         the model to be trained. Default: spatio temporal
                       transformer set by "spatio". Other options are "dvd" and "vdsr" for deep video deblurring and very deep super resolution method

  structure <br/>


Example usage <br/>
python main_spatio.py --cuda --batchSize 32 --lr 0.1 --dataset /path/to/training/data
--model vdsr

## Test
Test is the eval_loop.py file. It takes both input and ground truth images, processes the input image using selected network and calculate the PSNR between model output and ground image and between ground truth image and input image <br/>
Testing is done with the eval_loop.py file <br/>
usage: eval_loop.py [-h] [--cuda] [--model MODEL] [--dataset DATASET]
                    [--gpus GPUS] <br/>

optional arguments: <br/>
  -h, --help         show this help message and exit <br/>
  --cuda             use cuda? <br/>
  --model MODEL      model path <br/>
  --dataset DATASET  dataset name <br/>
  --gpus GPUS        gpu ids (default: 0) <br/>

Example usage <br/>
python eval_loop.py --cuda --model /path/to/model/file --dataset /path/to/test/data <br/>
