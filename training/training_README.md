# Training Custom LoRA Models

This directory contains resources for training custom LoRA models to create different cartoon styles.

## 📚 Contents

- `simpsonify_training_kohya.ipynb` - **Actual training notebook used in google collab** 
- `README.md` - This file
- `prepared_images` - some of the images I used for training 

## 🚀 Quick Start I did (Google Colab - Kohya Method)

**This is the actual training method I used for the models in this repository.**

### What is Kohya sd-scripts?

Kohya's training scripts are the industry-standard for LoRA training, offering:
- Advanced optimization techniques
- Noise offset and adaptive noise scaling
- Min-SNR gamma weighting
- Better convergence and quality
- More control over training parameters

### Prerequisites

- Google account (for Google Colab)
- Training dataset (images + captions)
- ~100-120 minutes training time

### Step 1: Open Notebook in Colab

1. Uploaded `simpsonify_training_kohya.ipynb` to Google Drive
2. Right-click → Open with → Google Colaboratory


### Step 2: Prepared Dataset

dataset should contain:
- **Images**: Portrait photos (any resolution, will be resized to 512x512)
- **Captions**: Text files describing each image

**Directory structure:**
```
dataset.zip
└── 20_simpsons/  (or any folder name)
    ├── image_001.png
    ├── image_001.txt
    ├── image_002.png
    ├── image_002.txt
    └── ...
```

**Caption format:**
```
cartoon_style, simpson, animated
```

**That was helpful:**
- Used consistent trigger words (e.g., `simpsons_style`, `cartoonify`)
- Kept captions simple and consistent
- 3-8 words is ideal
- All images should get the same caption for style training

### Step 3: Uploaded Dataset to Colab


### Step 4: Chose Training Configuration

The notebook provides two pre-configured setups since i trained twice:

**Config A: Simpsons Style (Basic)**
```python
- LoRA Rank: 16, Alpha: 16
- Epochs: 12
- Learning Rate: 1e-4
- Scheduler: constant
- Optimizer: AdamW
```

**Config B: Cartoonify Style (Advanced)**
```python
- LoRA Rank: 16, Alpha: 8
- Epochs: 10
- Learning Rate: 5e-5
- Scheduler: cosine_with_restarts
- Optimizer: AdamW8bit
- Extra: noise offset, min-SNR gamma
```

### Step 5: Ran Training

1. **Connected to GPU runtime:**
   - Runtime → Change runtime type → T4 GPU 

2. **Ran all cells** sequentially

### Step 6: Tested and Compared Checkpoints

The notebook includes cells to:
- Test different checkpoints 
- Generate comparison images
- See which epoch performs best

### Step 7: Downloaded Model

At the end of training:
1. Reviewed test images
2. Chose best checkpoint
3. Downloaded the `.safetensors` file



### Hyperparameter Tuning

**If results are blurry/weak:**
- Increase `LORA_RANK` to 32
- Increase `MAX_TRAIN_STEPS` to 5000
- Lower `LEARNING_RATE` to 5e-5

**If results are over-stylized/distorted:**
- Decrease `LORA_RANK` to 8
- Stop training earlier (~2000 steps)
- Increase `LORA_DROPOUT` to 0.1

**If training is too slow:**
- Use Colab Pro (V100 GPU)
- Reduce `RESOLUTION` to 448 or 384
- Reduce `MAX_TRAIN_STEPS`



## 💾 Saving Training Checkpoints

The notebook saves checkpoints every 500 steps:
```
output/lora/checkpoint-500/
output/lora/checkpoint-1000/
output/lora/checkpoint-1500/
...
```
