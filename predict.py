import os
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from datasets.dataset import JointTransform2D, correct_dims
from models.par_segnet import PARSegNet


class FetalDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

        images = [sitk.GetArrayFromImage(sitk.ReadImage(str(p))) for p in image_paths]
        labels = [sitk.GetArrayFromImage(sitk.ReadImage(str(p))) for p in label_paths]
        print(f"Raw image shapes: {[img.shape for img in images]}")
        print(f"Raw image min/max: {[img.min() for img in images]}, {[img.max() for img in images]}")
        print(f"Raw label shapes: {[lbl.shape for lbl in labels]}")
        print(f"Raw label min/max: {[lbl.min() for lbl in labels]}, {[lbl.max() for lbl in labels]}")
        self.images = np.array(images)
        self.labels = np.array(labels)
        print(f"Loaded {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, mask = correct_dims(self.images[idx].transpose(1, 2, 0), self.labels[idx])
        # print(f"Image shape after correct_dims: {image.shape}, Min/Max: {image.min()}, {image.max()}")
        # print(f"Mask shape after correct_dims: {mask.shape}, Min/Max: {mask.min()}, {mask.max()}")
        if self.transform:
            image, mask, low_mask = self.transform(image, mask)
        # print(f"Image after transform: {image.shape}, Min/Max: {image.min()}, {image.max()}")
        # print(f"Mask after transform: {mask.shape}, Min/Max: {mask.min()}, {mask.max()}")
        # print(f"Low_mask after transform: {low_mask.shape}, Min/Max: {low_mask.min()}, {low_mask.max()}")
        return {
            'image': image,
            'label': mask.unsqueeze(0),
            'low_mask': low_mask.unsqueeze(0)
        }

def main():
    device = torch.device('cuda')
    os.makedirs("./pred", exist_ok=True)
    os.makedirs("./gt", exist_ok=True)

    
    ckpt = "xxxx.pth"
    model = PARSegNet(
            in_ch=3,
            out_ch=3
        )
    state_dict = torch.load(f"./model/fold0/{ckpt}")
    model.load_state_dict(state_dict)
    print(f"Loaded model: {ckpt}")
    model.to(device)

    tf_val = JointTransform2D(img_size=256, low_img_size=128, ori_size=256,
                              crop=None, p_flip=0.0, color_jitter_params=None, long_mask=True)
    root_path = Path('./testdataset')

   
    image_files_list = sorted([f for f in os.listdir(root_path / "image_mha") if f.endswith('.mha')])
    start_idx = int(image_files_list[0].replace('.mha', '')) 
    end_idx = int(image_files_list[-1].replace('.mha', ''))  
    image_files = np.array([(root_path / Path("image_mha") / Path(str(i).zfill(5) + '.mha')) for i in range(start_idx, end_idx + 1)])
    label_files = np.array([(root_path / Path("label_mha") / Path(str(i).zfill(5) + '.mha')) for i in range(start_idx, end_idx + 1)])

    
    test_txt_path = "./test.txt"
    if not os.path.exists(test_txt_path):
        raise FileNotFoundError(f"Test file not found at: {test_txt_path}")
    with open(test_txt_path, "r") as file:
        lines = file.readlines()
    test_index = [int(line.strip()) for line in lines]  
    test_index = [i - start_idx for i in test_index if start_idx <= i <= end_idx]  
    print(f"Number of test samples: {len(test_index)}")
    db_test = FetalDataset(transform=tf_val,
                           image_paths=[image_files[i] for i in test_index],
                           label_paths=[label_files[i] for i in test_index])
    testloader = DataLoader(db_test, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    # dice_list = []

    with torch.no_grad():
        model.eval()

        for batch_idx, datapack in tqdm(enumerate(testloader)):
            imgs = datapack['image'].to(dtype=torch.float32, device='cuda')
            masks = datapack['label'].to(dtype=torch.float32, device='cuda')
            preds = model(imgs)

            for i in range(preds.shape[0]):
                # Obtain class-index masks: 0, 1, 2
                pred = (
                    preds[i]
                    .argmax(dim=0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                label = (
                    masks[i]
                    .squeeze(0)
                    .long()
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                # Use matching numeric filenames
                idx = batch_idx * testloader.batch_size + i + 1

                pred_path = f"./pred/{idx}.png"
                gt_path = f"./gt/{idx}.png"

                if pred.shape != label.shape:
                    raise ValueError(
                        f"Shape mismatch: pred={pred.shape}, gt={label.shape}"
                    )

                if not cv2.imwrite(pred_path, pred):
                    raise IOError(f"Failed to save prediction: {pred_path}")

                if not cv2.imwrite(gt_path, label):
                    raise IOError(f"Failed to save ground truth: {gt_path}")
                
       

if __name__ == '__main__':
    main()
