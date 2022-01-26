# MACN
Original PyTorch implementation of "Joint Learning of Video Scene Detection and Annotation via Multi-modal Adaptive Context Network" 

## Discription
### Data : 
Data sets in the form of MP4, npy, pkl and txt
### Label : 
Ground truth corresponding to Data folders.
### output: 
Results folder.
### inference.py: 
Inference code.
### model.py: 
The framework of MACN.
### requirement.txt:
Requirement documents.




## Requirements
The code was implemented using Python 3.8.8 and the following packages:
```
apex==0.1
ffmpeg-python==0.2.0
opencv-python==4.5.4.58
timm==0.4.12
tokenizers==0.10.3
torch==1.10.0
torchaudio==0.10.0
torchvision==0.11.1
```
## Feature extraction
visual: [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

audio: [VGGish](https://github.com/harritaylor/torchvggish)

More hyperparameters and  details can be found in our paper.

## Model weights
For convenience, we provide our model weights on [google drive](https://drive.google.com/file/d/1pXibHeL_YJeMGekJBGJQh8OELMdoi-5z/view?usp=sharing). You need download it and put it in the same directory as inference.py

## Testing MACN
To test the experiments from the paper, please execute the following:

```
python inference.py
```
