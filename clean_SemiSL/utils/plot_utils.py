import numpy as np
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import math

NO_LABEL = -1
N_COLS = 4

def plot_batch(
    input_batch: torch.Tensor,
    pred_batch: torch.Tensor,
    label_batch: torch.Tensor,
    task_type: str,
    save: str,
    **kwargs
):
    n = len(input_batch)

    if task_type == "image_classification":
        ncols, nrows = N_COLS, math.ceil(n/N_COLS)
        plt.figure(figsize=(8*nrows, 8*ncols))
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False)
        
        for i in range(ncols*nrows):
            col = i%ncols
            row = i//ncols
            axs[row,col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            
            if i >= len(input_batch): continue

            img = input_batch[i].detach()
            img = F.to_pil_image(img)

            if label_batch is not None:
                if label_batch[i] != -1:
                    y = int(label_batch[i].item()) 
                    cls = str(y) if not "class_names" in kwargs else kwargs["class_names"][y]
                    axs[row,col].set_title(f"Label = {cls}")
                else:
                    axs[row,col].set_title(f"Unlabelled")

            axs[row,col].imshow(np.asarray(img))
 
    else:
        raise NotImplementedError

    if save:
        plt.savefig(save)
    else:
        return fig
