# Spatio-Temporal Transformer Networkfor Video Restoration
This is implementation of the paper [Spatio-Temporal Transformer Networkfor Video Restoration](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwiOq8qu-eHiAhXIlIsKHbdGCHwQFjAAegQIAhAC&url=http%3A%2F%2Fopenaccess.thecvf.com%2Fcontent_ECCV_2018%2Fpapers%2FTae_Hyun_Kim_Spatio-temporal_Transformer_Network_ECCV_2018_paper.pdf&usg=AOvVaw0lhDjBkIQbpuxCIE3k9a0Q)
## Dependencies
The code depends on pytorch and pillow.

## Training
For training you need to call main_spatio.py file with the corresponding option parameters.
Example usage
python main_spatio.py --cuda --batchSize 32 --lr 0.1 --dataset /path/to/training/data

##Testing
Testing is done with the eval_loop.py file 
Example usage
python eval_loop.py --cuda --model /path/to/model/file --dataset /path/to/test/data
