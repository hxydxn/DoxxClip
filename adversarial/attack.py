# %% [markdown]
# This code was based on a highly modified version of the script linked below
# # Tutorial 10: Adversarial attacks

# %% [markdown]
# ![Status](https://img.shields.io/static/v1.svg?label=Status&message=Finished&color=green)
# 
# **Filled notebook:**
# [![View notebooks on Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=View%20On%20Github&color=lightgrey)](https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial10/Adversarial_Attacks.ipynb)
# [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial10/Adversarial_Attacks.ipynb)  
# **Pre-trained models and dataset:**
# [![View files on Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=View%20On%20Github&color=lightgrey)](https://github.com/phlippe/saved_models/tree/main/tutorial10)
# [![GoogleDrive](https://img.shields.io/static/v1.svg?logo=google-drive&logoColor=yellow&label=GDrive&message=Download&color=yellow)](https://drive.google.com/drive/folders/1k01P6w31VOW9TT0gTEP9kog315qyiXCd?usp=sharing)  
# **Recordings:**
# [![YouTube - Part 1](https://img.shields.io/static/v1.svg?logo=youtube&label=YouTube&message=Part%201&color=red)](https://youtu.be/uidLtkhZFwY)
# [![YouTube - Part 2](https://img.shields.io/static/v1.svg?logo=youtube&label=YouTube&message=Part%202&color=red)](https://youtu.be/Dmbz0ffc6Wg)
# [![YouTube - Part 3](https://img.shields.io/static/v1.svg?logo=youtube&label=YouTube&message=Part%203&color=red)](https://youtu.be/0dt2Su-SRpI)    
# **Author:** Phillip Lippe

# %%
## Standard libraries
import os
import json
import math
import time
import numpy as np
import scipy.linalg

## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.set()

## Progress bar
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Path to the folder where the datasets are/should be downloaded (e.g. MNIST)
DATASET_PATH = "data/img_resized_1M/cities_instagram/"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "saved_models/tutorial10"

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Fetching the device that will be used throughout this notebook
torch.cuda.empty_cache()
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:3")
print("Using device", device)
from torchvision.datasets import ImageFolder
data_dir = "./data/img_resized_1M/cities_instagram/"
dataset = ImageFolder(data_dir)

# %%
# Load CNN architecture pretrained on ImageNet
from transformers import AutoModelForZeroShotImageClassification
os.environ["TORCH_HOME"] = CHECKPOINT_PATH
from transformers import CLIPImageProcessor, CLIPProcessor

label2id, id2label = dict(), dict()
for i, label in enumerate(dataset.classes):
    label2id[label] = i
    id2label[i] = label
pretrained_model = AutoModelForZeroShotImageClassification.from_pretrained(
    "geolocal/StreetCLIP",
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True,
)
pretrained_model.to(device)
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
processor = processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
# replace this with the hf calls to load the model. I think it will be 1:1

# No gradients needed for the network
pretrained_model.eval()
for p in pretrained_model.parameters():
    p.requires_grad = False

# %% [markdown]
# To perform adversarial attacks, we also need a dataset to work on. Given that the CNN model has been trained on ImageNet, it is only fair to perform the attacks on data from ImageNet. For this, we provide a small set of pre-processed images from the original ImageNet dataset (note that this dataset is shared under the same [license](http://image-net.org/download-faq) as the original ImageNet dataset). Specifically, we have 5 images for each of the 1000 labels of the dataset. We can load the data below, and create a corresponding data loader.

# %%

train_idx = np.load("train_index.npy").tolist()
test_idx = np.load("test_index.npy").tolist()



# %%
# %%
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

def collate_fn(batch):
    batched = []
    for i, (image, label) in enumerate(batch):
        batched.append({'pixel_values': image.to(device), 'attention_mask': torch.ones(len(batch), 1, 224, 224).to(device)})
    return batched
resize = Resize([336,336])
        
train_transforms = Compose(
        [
            resize,
            ToTensor(),
            normalize,
            
        ]
    )
dataset = ImageFolder(data_dir, transform=train_transforms)


# %%
data_loader = data.DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=1, shuffle=False, drop_last=False, num_workers=1)
label_names = dataset.classes

def get_label_index(lab_str):
    assert lab_str in label_names, f"Label \"{lab_str}\" not found. Check the spelling of the class."
    return label_names.index(lab_str)

# %% [markdown]
# Before we start with our attacks, we should verify the performance of our model. As ImageNet has 1000 classes, simply looking at the accuracy is not sufficient to tell the performance of a model. Imagine a model that always predicts the true label as the second-highest class in its softmax output. Although we would say it recognizes the object in the image, it achieves an accuracy of 0. In ImageNet with 1000 classes, there is not always one clear label we can assign an image to. This is why for image classifications over so many classes, a common alternative metric is "Top-5 accuracy", which tells us how many times the true label has been within the 5 most-likely predictions of the model. As models usually perform quite well on those, we report the error (1 - accuracy) instead of the accuracy:

# %%
encodings = [[49406,  3530, 49407],
        [49406,  1789, 49407],
        [49406, 14638, 49407],
        [49406,  6995, 49407],
        [49406,  4891, 49407],
        [49406, 10454, 49407],
        [49406, 20254, 49407],
        [49406,  6754, 49407],
        [49406,  5278, 49407],
        [49406,  4514, 49407]]

encodings = torch.tensor(encodings).to(device)

def eval_model(dataset_loader, img_func=None):
    tp, tp_5, counter = 0., 0., 0.
    i=0
    for imgs, labels in dataset_loader:
        i+=1
        imgs = imgs.to(device)
        labels = labels.to(device)
        input = {'pixel_values': imgs, 'attention_mask': torch.ones(encodings.shape).to(device), 'input_ids': encodings}
        with torch.no_grad():
            outputs = pretrained_model(**input)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            preds = logits_per_image.softmax(dim=1)
        tp += (preds.argmax(dim=-1) == labels).sum()
        tp_5 += (preds.topk(5, dim=-1)[1] == labels[...,None]).any(dim=-1).sum()
        counter += preds.shape[0]
        if i == 1000:
            break
    acc = tp.float().item()/counter
    top5 = tp_5.float().item()/counter
    print(f"Top-1 error: {(100.0 * (1 - acc)):4.2f}%")
    print(f"Top-5 error: {(100.0 * (1 - top5)):4.2f}%")
    return acc, top5


# %%
def min_max_normalize(img):
    v_min, v_max = img.min(), img.max()
    v_p = (img - v_min)/(v_max - v_min)
    return v_p

# %%
NORM_STD = torch.tensor(image_processor.image_std).view(3,1,1).numpy()[0][0]
NORM_MEAN = torch.tensor(image_processor.image_mean).view(3,1,1).numpy()[0][0]

def show_prediction(img, label, pred, K=5, adv_img=None, noise=None):

    if isinstance(img, torch.Tensor):
        # Tensor image to numpy
        img = img.cpu().permute(1, 2, 0).numpy()
        img = (img * NORM_STD[None,None]) + NORM_MEAN[None,None]
        img = np.clip(img, a_min=0.0, a_max=1.0)
        if isinstance(label, torch.Tensor) or isinstance(label, np.ndarray):
                      label = label.item()

    # Plot on the left the image with the true label as title.
    # On the right, have a horizontal bar plot with the top k predictions including probabilities
    if noise is None or adv_img is None:
        fig, ax = plt.subplots(1, 2, figsize=(10,2), gridspec_kw={'width_ratios': [1, 1]})
    else:
        fig, ax = plt.subplots(1, 5, figsize=(12,2), gridspec_kw={'width_ratios': [1, 1, 1, 1, 2]})

    ax[0].imshow(min_max_normalize(img))
    ax[0].set_title(label_names[label])
    ax[0].axis('off')

    if adv_img is not None and noise is not None:
        # Visualize adversarial images
        adv_img = adv_img.cpu().permute(1, 2, 0).numpy()
        adv_img = np.clip(adv_img, a_min=0.0, a_max=1.0)
        ax[1].imshow(adv_img)
        ax[1].set_title('Adversarial')
        ax[1].axis('off')
        # Visualize noise
        noise = min_max_normalize(noise.cpu()).permute(1, 2, 0).numpy()
        ax[2].imshow(noise)
        ax[2].set_title('Noise')
        ax[2].axis('off')
        # buffer
        ax[3].axis('off')

    if abs(pred.sum().item() - 1.0) > 1e-4:
        pred = torch.softmax(pred, dim=-1)
    topk_vals, topk_idx = pred.topk(K, dim=-1)
    topk_vals, topk_idx = topk_vals.cpu().numpy(), topk_idx.cpu().numpy()
    topk_idx = topk_idx[0]
    topk_vals = topk_vals[0]
    ax[-1].barh(np.arange(K), topk_vals*100.0, align='center', color=["C0" if topk_idx[i]!=label else "C2" for i in range(K)])
    ax[-1].set_yticks(np.arange(K))
    ax[-1].set_yticklabels([label_names[c] for c in topk_idx])
    ax[-1].invert_yaxis()
    ax[-1].set_xlabel('Confidence')
    ax[-1].set_title('Predictions')

    plt.show()
    plt.close()

# %% [markdown]
# Let's visualize a few images below:

# %%
exmp_batch, label_batch = next(iter(data_loader))
input = {'pixel_values': exmp_batch.to(device), 'attention_mask': torch.ones(encodings.shape).to(device), 'input_ids': encodings}
with torch.no_grad():
    outputs = pretrained_model(**input)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    preds = logits_per_image.softmax(dim=1)
    # show_prediction(exmp_batch[0], label_batch[0], preds)

# %%
def show_noise(exmp_batch, label, noise): 
    input = {'pixel_values': exmp_batch.to(device), 'attention_mask': torch.ones(encodings.shape).to(device), 'input_ids': encodings}
    with torch.no_grad():
        outputs = pretrained_model(**input)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        preds = logits_per_image.softmax(dim=1)
        # show_prediction(exmp_batch[0], label, preds)
    return preds

# %%
def call_box(exmp_batch): 
    input = {'pixel_values': exmp_batch.to(device), 'attention_mask': torch.ones(encodings.shape).to(device), 'input_ids': encodings}
    with torch.no_grad():
        outputs = pretrained_model(**input)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        preds = logits_per_image.softmax(dim=1)
        # show_prediction(exmp_batch[0], label_batch[0], preds)
    return preds

# %% [markdown]
# 16 chunks. at each iteration select the chunk with the worst loss and update, storing the values already explored

# %%
class bandit_mask():
    def __init__(self, init_pixels, device):
        self.device = device
        self.init_pixels = init_pixels
        self.mask = torch.zeros((1,3,336,336)).to(device)
        self.last_mask = torch.zeros((1,3,336,336)).to(device)
        self.chunk_loss = [0 for i in range(36)]
        self.pulls = []
        self.arms = [self.new_arm(arm_id=i) for i in range(36)]
        self.last_pull = 0
        for i in range(36):
            self.update_mask(i)
        self.pulls = [np.random.randint(36)]
        self.rewards = [5 for i in range(36)]
        
        self.last_loss = 10
        self.best_5_masks = [torch.zeros((1,3,336,336)) for i in range(5)]
        self.best_loss = torch.Tensor([10,10,10,10,10]).to(device)
        self.cum_loss = 0
        # if pulling an arm is successful, we want to leave that arm alone

    def __call__(self, picture, mask_ = None):
        if mask_ == None:
            mask_ = self.mask
        picture = torch.nn.functional.normalize(picture)
        picture = min_max_normalize(picture)
        masked = picture + mask_

        return torch.clamp(masked, 0 ,1)
    
    def new_arm(self, arm_id=None):
        if arm_id != None:
            self.pulls.append(arm_id)
        
        arm = torch.zeros([self.init_pixels,3]).to(device)
        for i in range(self.init_pixels):
            arm[i,0] = np.random.randint(3)
            arm[i,1] = np.random.randint(56)
            arm[i,2] = np.random.randint(56)
        return arm
    
    def update_mask(self, arm_id):
        for j in range(self.init_pixels):
                self.mask[0, int(self.arms[arm_id][j][0].item()), ((arm_id)%6)*56 + int(self.arms[arm_id][j][1].item()), int((arm_id - (arm_id)%6)/6)*56+int(self.arms[arm_id][j][2].item())] = 999 if np.random.randint(2) else -999

    def play(self, loss):
        if loss < self.last_loss: # If the pull was strictly better than the last one
            self.rewards[self.pulls[-1]] += 3
            if loss < max(self.best_loss):
                self.best_5_masks[torch.argmax(self.best_loss)] = self.mask
                self.best_loss[torch.argmax(self.best_loss)] = loss
            # pull again
            self.cum_loss += loss
            self.last_mask = self.mask
            self.last_pull = self.arms[self.pulls[-1]]
            next_options = self.rewards.copy()
            next_options[self.pulls[-1]]=0
            self.pulls.append(np.argmax(next_options))
            self.arms[self.pulls[-1]] = self.new_arm(arm_id=self.pulls[-1])    
            self.pulls.append(self.pulls[-1])
            self.update_mask(self.pulls[-1])
            self.last_loss = loss 
        elif loss < self.cum_loss / len(self.pulls): # If the pull is better than the running avg, but not the last pull
            self.rewards[self.pulls[-1]] += 2 # 1.5 the reward of the previous pull
            self.cum_loss += loss
            if loss < max(self.best_loss):
                self.best_5_masks[torch.argmax(self.best_loss)] = self.mask
                self.best_loss[torch.argmax(self.best_loss)] = loss
            # always pull from a different arm - encourages exploration
            self.last_mask = self.mask
            self.last_pull = self.arms[self.pulls[-1]]
            next_options = self.rewards.copy()
            next_options[self.pulls[-1]] = 0
             
            self.pulls.append(np.argmax(next_options))
            self.arms[self.pulls[-1]] = self.new_arm(arm_id=self.pulls[-1])
            self.update_mask(self.pulls[-1])
            self.last_loss = loss 
        else: # undo last play and pull another arm
            self.rewards[self.pulls[-1]] -= 3
            self.mask = self.last_mask
            self.arms[self.pulls[-1]] = self.last_pull
            next_options = self.rewards.copy()
            next_options[self.pulls[-1]] = 0
            self.pulls[-1] = np.argmax(next_options) # If the pull is worse than the running avg, undo the pull and pull again   
            self.arms[self.pulls[-1]] = self.new_arm(arm_id=self.pulls[-1])
            self.update_mask(self.pulls[-1])
            if max(self.rewards) <= 0:
                print("Arms could not find any rewards, Rescaling")
                self.mask = min_max_normalize(self.mask)
                self.rewards = [3 for i in range(len(self.rewards))]
                return False
            
        return True        



# %%
from matplotlib import pyplot as plt
n = 10000
top_n_correct = np.load("top_" + str(n) + "_correct.npy")
top_n_index = top_n_correct[:,10]
test_idx = np.load("test_index.npy")
epochs = 60
REGULARIZATION = 0.1
SUBSET_LENGTH = 8000
BATCH_SIZE = 16
LEARNING_RATE = 0.001
mask = bandit_mask(init_pixels=100, device=device)
test_dir = 3


# %%

for epoch in range(epochs):
    ideal = torch.tensor([0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11]).to(device)
    random_subset = [top_n_correct[np.random.randint(0,n),:]  for i in range(SUBSET_LENGTH)] # bagging
    loss_buffer = torch.zeros([BATCH_SIZE, ideal.shape[0]], dtype=float).to(device)
    label_buffer = torch.zeros([BATCH_SIZE], dtype=int).to(device)
    pic_buff = torch.zeros([1,3,336,336], dtype=float).to(device)
    batch = 0
    sample = 0
    falses = 0
    for i in range(len(random_subset)):
        index = int(random_subset[i][10])
        picture, label = data_loader.dataset[index]
        
        picture = torch.unsqueeze(picture, 0).to(device)
        pic_buff += picture
        tensor_image = mask(picture)
        
        preds = call_box(tensor_image)
        ideal = torch.tensor(random_subset[i][0:10]).to(device)
        
        loss = torch.nn.functional.mse_loss(preds[0], ideal)
        loss_buffer[sample] = loss
        label_buffer[sample] = label
        sample += 1
        if sample == BATCH_SIZE - 1:
            sample = 0
            batch += 1
            pic_buff = pic_buff / BATCH_SIZE
            loss = torch.mean(loss_buffer)
            if batch % 32 == 0:
                print("Batch", batch, "Loss:", loss.item())
                print(mask.rewards)
                print(mask.cum_loss / len(mask.pulls))
            # if batch % 64 == 0:
                
                # show_prediction(picture[0], label, preds, adv_img=tensor_image[0], noise=mask.mask[0])
                
            if(not mask.play(loss)):
               falses += 1
               print("Falses: ", falses)
               if falses > 16:
                   break
            pic_buff = torch.zeros([1,3,336,336], dtype=float).to(device)
    
    if epoch % 5 == 0 or falses > 16:
        for i, masks in enumerate(mask.best_5_masks):

            np.save("test_" + str(test_dir) + "/mask_" + str(i) + "_epoch_" + str(epoch) + ".npy", masks.cpu().numpy())
            # show_prediction(picture[0], label, preds, adv_img=mask(picture, masks)[0], noise=torch.clamp(masks[0],0,1))
    if falses > 16:
        break
print("Done")


