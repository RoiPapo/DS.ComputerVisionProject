import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

relevant_folders = ['P016_balloon1_side', 'P016_balloon2_side', 'P016_tissue1_side', 'P016_tissue2_side',
                    'P017_balloon1_side', 'P017_balloon2_side', 'P017_tissue1_side', 'P017_tissue2_side',
                    'P018_balloon1_side', 'P018_balloon2_side', 'P018_tissue1_side', 'P018_tissue2_side',
                    'P019_balloon1_side', 'P019_balloon2_side', 'P019_tissue1_side', 'P019_tissue2_side',
                    'P020_balloon1_side', 'P020_balloon2_side', 'P020_tissue1_side', 'P020_tissue2_side',
                    'P021_balloon1_side', 'P021_balloon2_side', 'P021_tissue1_side', 'P021_tissue2_side',
                    'P022_balloon1_side', 'P022_balloon2_side', 'P022_tissue1_side', 'P022_tissue2_side',
                    'P023_balloon1_side', 'P023_balloon2_side', 'P023_tissue1_side', 'P023_tissue2_side',
                    'P024_balloon1_side', 'P024_balloon2_side', 'P024_tissue1_side', 'P024_tissue2_side',
                    'P025_balloon1_side', 'P025_balloon2_side', 'P025_tissue1_side', 'P025_tissue2_side',
                    'P026_balloon1_side', 'P026_balloon2_side', 'P026_tissue1_side', 'P026_tissue2_side',
                    'P027_balloon1_side', 'P027_balloon2_side', 'P027_tissue1_side', 'P027_tissue2_side',
                    'P028_balloon1_side', 'P028_balloon2_side', 'P028_tissue1_side', 'P028_tissue2_side',
                    'P029_balloon1_side', 'P029_balloon2_side', 'P029_tissue1_side', 'P029_tissue2_side',
                    'P030_balloon1_side', 'P030_balloon2_side', 'P030_tissue1_side', 'P030_tissue2_side',
                    'P031_balloon1_side', 'P031_balloon2_side', 'P031_tissue1_side', 'P031_tissue2_side',
                    'P032_balloon1_side', 'P032_balloon2_side', 'P032_tissue1_side', 'P032_tissue2_side',
                    'P033_balloon1_side', 'P033_balloon2_side', 'P033_tissue1_side', 'P033_tissue2_side',
                    'P034_balloon1_side', 'P034_balloon2_side', 'P034_tissue1_side', 'P034_tissue2_side',
                    'P035_balloon1_side', 'P035_balloon2_side', 'P035_tissue1_side', 'P035_tissue2_side',
                    'P036_balloon1_side', 'P036_balloon2_side', 'P036_tissue1_side', 'P036_tissue2_side',
                    'P037_balloon1_side', 'P037_balloon2_side', 'P037_tissue1_side', 'P037_tissue2_side',
                    'P038_balloon1_side', 'P038_balloon2_side', 'P038_tissue1_side', 'P038_tissue2_side',
                    'P039_balloon1_side', 'P039_balloon2_side', 'P039_tissue1_side', 'P039_tissue2_side',
                    'P040_balloon1_side', 'P040_balloon2_side', 'P040_tissue1_side', 'P040_tissue2_side']


# Define EfficientNet feature extractor
class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(EfficientNetFeatureExtractor, self).__init__()
        self.efficientnet = efficientnet
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.efficientnet.extract_features(x)['features']
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x


# Define dataset and dataloader
class ImageDataset(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[152.33031877955463, 106.26509461301819, 104.55854576464021],
                                 std=[36.21155284418057, 30.75150171154211, 31.2230456008511])
        ])
        self.samples = []
        for filename in sorted(os.listdir(root)):
            self.samples.append((os.path.join(root, filename), 0))

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)


def extract_folder(feature_extractor, folder, data_folder, dest_path, batch_size):
    dataset = ImageDataset(data_folder)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Create feature extractor and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor.to(device)

    # Extract features from images and save to file
    features = []
    labels = []
    feature_extractor.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting features'):
            images, batch_labels = batch
            images = images.to(device)
            features.append(feature_extractor(images).cpu().numpy())
            labels.append(batch_labels.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    os.makedirs(dest_path, exist_ok=True)
    np.save(f'{dest_path}/{folder}.npy', features)
    # np.save('labels.npy', labels)


def extract_features(db_address, batch_size, seed, dest_path):
    for model_num in range(8):
        model = EfficientNet.from_pretrained(f'efficientnet-b{model_num}')
        for folder_vid in relevant_folders:
            print(f'Extracting features for efficientnet b{model_num}')
            extract_folder(feature_extractor=model,
                           folder=folder_vid,
                           data_folder=f'{db_address}/{folder_vid}',
                           dest_path=f'{dest_path}/efficientnet/b{model_num}',
                           batch_size=24)


def main():
    db_address = f'/home/user/datasets/frames'

    # data_folder = f'/datashare/APAS/frames'

    dest_path = f'/home/user/test'

    extract_features(db_address=db_address,
                     batch_size=24,
                     seed=100,
                     dest_path=dest_path)


if __name__ == '__main__':
    main()
