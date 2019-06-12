# Spatio-Temporal Transformer Network for Video Restoration
This is implementation of the paper [Spatio-Temporal Transformer Networkfor Video Restoration](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwiOq8qu-eHiAhXIlIsKHbdGCHwQFjAAegQIAhAC&url=http%3A%2F%2Fopenaccess.thecvf.com%2Fcontent_ECCV_2018%2Fpapers%2FTae_Hyun_Kim_Spatio-temporal_Transformer_Network_ECCV_2018_paper.pdf&usg=AOvVaw0lhDjBkIQbpuxCIE3k9a0Q)
## Dependencies
The code was developed on python3 with pytorch and pillow libraries. Please visit [installation guide](https://pytorch.org/get-started/locally/) for pytorch installation. For installing the pillow simple type <br/>
`pip3 install pillow`
on terminal

## Dataset
The code was trained on [Deep Video Deblurring](https://arxiv.org/pdf/1611.08387)'s
dataset which can be accessed from this [link](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset.zip). Unzip it into a desired
folder. Alternatively, you can place your own videos under <br/>
`dataset/qualitative_datasets/[video_file_name]/input` as input and <br/>
`dataset/qualitative_datasets/[video_file_name]/GT` as ground truth videos <br/>
as frame extracted videos. You can extract a video into frames using ffmpeg with <br/>
the following command <br/>
`ffmpeg -i file.mpg -r 1/1 $foldername/%04d.jpg` <br/>
where `$foldername` is desired folder for frame extraction

## Training
For training you need to call main_spatio.py file with the corresponding option parameters.
usage: main_spatio.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS]
                      [--lr LR] [--step STEP] [--cuda] [--resume RESUME]
                      [--start-epoch START_EPOCH] [--threads THREADS]
                      [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
                      [--pretrained PRETRAINED] [--gpus GPUS]
                      [--dataset DATASET]


optional arguments:

  --batchSize BATCHSIZE Training batch size
  --nEpochs NEPOCHS     Number of epochs to train for
  --lr LR               Learning Rate. Default=0.1
  --step STEP           Sets the learning rate to the initial LR decayed by
                        momentum every n epochs, Default: n=10
  --cuda                Use cuda?
  --resume RESUME       Path to checkpoint (default: none)
  --start-epoch START_EPOCH
                        Manual epoch number (useful on restarts)
  --threads THREADS     Number of threads for data loader to use, Default: 1
  --momentum MOMENTUM   Momentum, Default: 0.9
  --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        Weight decay, Default: 1e-4
  --pretrained PRETRAINED
                        path to pretrained model (default: none)
  --gpus GPUS           gpu ids (default: 0)
  --dataset DATASET     the folder where dataset can be found with specified
  structure


Example usage
python main_spatio.py --cuda --batchSize 32 --lr 0.1 --dataset /path/to/training/data

## Test
Testing is done with the eval_loop.py file
usage: eval_loop.py [-h] [--cuda] [--model MODEL] [--dataset DATASET]
                    [--gpus GPUS]

optional arguments:
  -h, --help         show this help message and exit
  --cuda             use cuda?
  --model MODEL      model path
  --dataset DATASET  dataset name
  --gpus GPUS        gpu ids (default: 0)

Example usage
python eval_loop.py --cuda --model /path/to/model/file --dataset /path/to/test/data
