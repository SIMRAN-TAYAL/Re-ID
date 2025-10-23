# Vehicle Re-Identification (Re-ID)

A deep learning-based Vehicle Re-Identification system that matches vehicles across non-overlapping camera views. This project implements and trains models to learn discriminative features for identifying the same vehicle in different images, a critical task in intelligent transportation systems and security.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
git clone https://github.com/SIMRAN-TAYAL/Re-ID.git
cd Re-ID
```

### 2. Get the Dataset
Download from: [VeRi Dataset on Kaggle](https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset)

Organize the dataset like this:
```
data/VeRi/
├── image_train/           # Training images
├── image_test/            # Testing images  
├── image_query/         # Query image names
├── train_label.xml        # Training labels
└── test_label.xml         # Testing labels
```

### 3. Train the Model
Open and run the notebook:
```bash
jupyter notebook notebooks/main.ipynb
```
Run cells 1-26 to train the model.

### 4. Evaluate the Model
Run cells 27+ to see performance results.

## 📁 Project Structure

```
Re-ID/
├── notebooks/
│   └── main.ipynb              # Main training and evaluation notebook
├── src/
│   ├── dataset.py              # VeRiDataset implementation
│   ├── sampler.py              # RandomIdentitySampler
│   ├── model.py                # ReIDModel with ResNet50 backbone
│   └── losses.py               # Triplet Loss implementation
├── data/
│   └── VeRi/                   # Dataset directory
├── Outputs/
│   ├── checkpoints/
│   │   └── ReID_model.pth      # Trained model weights
│   └── cached_embeddings/      # Feature cache for evaluation
└── README.md
```

## Model Details

- **Base Model**: ResNet50
- **Feature Size**: 512 dimensions
- **Training**: 60 epochs
- **Loss**: Triplet loss

## Dataset Info

- **Training**: 37,746 images (575 vehicles)
- **Testing**: 11,579 images (200 vehicles)
- **Query Images**: 1,678
- **Gallery Images**: 9,901

## How to Use

### Load trained model:
```python
from src.model import ReIDModel
import torch

model = ReIDModel(num_classes=575)
model.load_state_dict(torch.load('Outputs/checkpoints/ReID_model.pth'))
model.eval()

# Get features from vehicle image
features = model(image_tensor, return_feature=True)
```

## Evaluation Metrics & Results on VeRi Dataset

| Metric        | Description / Value                                      |
|---------------|----------------------------------------------------------|
| **mAP**       | Overall matching accuracy: **62.69%**                   |
| **Rank-1**    | Whether the vehicle is found in top 1 result: **87.84%**|
| **Rank-5**    | Whether the vehicle is found in top 5 results: **95.05%**|
| **Rank-10**   | Whether the vehicle is found in top 10 results: **97.44%**|
| **CMC Curve** | Visual representation of retrieval performance          |

