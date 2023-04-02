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
efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
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
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
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

def extract_folder(folder):
    # P016_balloon1_side
    # Load data and create dataloader
    data_folder = f'/home/user/datasets/frames/{folder}'
    # data_folder = f'/home/user/datasets/frames/{folder}'
    dest_path = f'{os.getcwd()}/efficientnet/B_exclude{0}'

    dataset = ImageDataset(data_folder)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=False)

    # Create feature extractor and move to device
    feature_extractor = EfficientNetFeatureExtractor()
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
    np.save(f'{dest_path}/{folder}.npy', features)
    # np.save('labels.npy', labels)


if __name__ == '__main__':
    extract_folder('P016_balloon1_side')