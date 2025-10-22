# -*- coding: utf-8 -*-
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from lxml import etree

class VeRiDataset(Dataset):
    """
    Vehicle Re-Identification Dataset Loader (for VeRi-776)
    Loads images and remaps vehicle IDs to contiguous integer labels.
    """

    def __init__(self, root_dir, split="train", transform=None, verbose=True):
        """
        Args:
            root_dir (str): Path to 'VeRi' folder.
            split (str): One of ['train', 'test', 'query'].
            transform: torchvision transforms.
            verbose (bool): If True, prints dataset statistics.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Select image directory and XML file based on split
        if split == "train":
            self.img_dir = os.path.join(root_dir, "image_train")
            label_file = os.path.join(root_dir, "train_label.xml")
        elif split == "test":
            self.img_dir = os.path.join(root_dir, "image_test")
            label_file = os.path.join(root_dir, "test_label.xml")
        elif split == "query":
            # usually query images are inside image_query folder
            self.img_dir = os.path.join(root_dir, "image_query")
            label_file = os.path.join(root_dir, "test_label.xml")
        else:
            raise ValueError(f"Invalid split name '{split}'")

        # Parse the XML file
        self.samples, self.vid_to_label = self._parse_xml(label_file)

        # Basic statistics
        if verbose:
            vids = [v for _, v, _ in self.samples]
            camids = [c for _, _, c in self.samples]
            print(f"\nLoaded VeRi split: {split}")
            print(f"  Total images: {len(self.samples)}")
            print(f"  Unique vehicle IDs: {len(set(vids))}")
            print(f"  Cameras: {len(set(camids))}")
            print(f"  Vehicle ID range (remapped): 0 → {len(set(vids))-1}\n")

    def _parse_xml(self, label_path):
        """
        Parses the XML label file, remaps vehicle IDs to contiguous indices.
        Returns:
            samples: list of (img_path, label, camid)
            vid_to_label: dict mapping original VID -> contiguous label
        """
        parser = etree.XMLParser(encoding='gb2312', recover=True)
        tree = etree.parse(label_path, parser=parser)
        root = tree.getroot()

        # Collect all unique VIDs that appear in this split
        vids = sorted(list({int(item.get("vehicleID")) for item in root.findall(".//Item")}))
        vid_to_label = {vid: idx for idx, vid in enumerate(vids)}

        samples = []
        for item in root.findall(".//Item"):
            img_name = item.get("imageName")
            vid = int(item.get("vehicleID"))
            camid = item.get("cameraID")
            camid = int(camid.replace("c", "")) if camid else 0

            img_path = os.path.join(self.img_dir, img_name)

            # Map original VID → contiguous label
            label = vid_to_label[vid]
            samples.append((img_path, label, camid))

        return samples, vid_to_label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, vid, camid = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, vid, camid
