import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_labeled_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    labels = []
    with open(file_path, 'r') as f:
        is_data_section = False
        for line in f:
            if line.startswith('DATA'):
                is_data_section = True
                continue
            if is_data_section:
                parts = line.split()
                if len(parts) == 4:
                    labels.append(int(parts[3]))
    labels = np.array(labels)
    return points, labels

def extract_cuboids(points, labels):
    cuboids = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == 0:
            continue
        label_points = points[labels == label]
        min_bound = np.min(label_points, axis=0)
        max_bound = np.max(label_points, axis=0)
        cuboid = np.hstack((min_bound, max_bound))
        cuboids.append(cuboid)
    return np.array(cuboids)

class PointCloudDetectionDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.data = []
        for file_path in self.file_paths:
            points, labels = load_labeled_pcd(file_path)
            cuboids = extract_cuboids(points, labels)
            self.data.append((points, labels, cuboids))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        points, labels, cuboids = self.data[idx]
        return points, labels, cuboids

def detection_collate_fn(batch):
    max_num_points = max([len(points) for points, _, _ in batch])
    point_clouds, labels_list, bbox_targets = [], [], []

    for points, labels, cuboids in batch:
        num_points = len(points)
        padded_points = np.pad(points, ((0, max_num_points - num_points), (0, 0)), mode='constant')
        point_clouds.append(padded_points)

        obj_labels = np.zeros(num_points)
        bbox_label = np.zeros((num_points, 6))

        for cuboid in cuboids:
            min_bound = cuboid[:3]
            max_bound = cuboid[3:]
            center = (min_bound + max_bound) / 2
            size = max_bound - min_bound
            in_bbox = np.all((points >= min_bound) & (points <= max_bound), axis=1)
            obj_labels[in_bbox] = 1
            bbox_label[in_bbox, :3] = center - points[in_bbox]
            bbox_label[in_bbox, 3:] = size

        obj_labels = np.pad(obj_labels, (0, max_num_points - num_points), mode='constant')
        bbox_label = np.pad(bbox_label, ((0, max_num_points - num_points), (0, 0)), mode='constant')

        labels_list.append(obj_labels)
        bbox_targets.append(bbox_label)

    point_clouds = torch.tensor(point_clouds, dtype=torch.float32).permute(0, 2, 1)
    labels_list = torch.tensor(labels_list, dtype=torch.float32)
    bbox_targets = torch.tensor(bbox_targets, dtype=torch.float32)
    return point_clouds, labels_list, bbox_targets

class PointNetDetector(nn.Module):
    def __init__(self):
        super(PointNetDetector, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)
        self.conv_obj = nn.Conv1d(1024, 1, 1)
        self.conv_bbox = nn.Conv1d(1024, 6, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, _, num_points = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        obj_scores = self.sigmoid(self.conv_obj(x))
        bbox_preds = self.conv_bbox(x)
        return obj_scores.squeeze(1), bbox_preds.permute(0, 2, 1)

def train_pointnet_detector(pcd_file_paths, epochs=500, batch_size=1, lr=0.0001):
    dataset = PointCloudDetectionDataset(pcd_file_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=detection_collate_fn)

    model = PointNetDetector().to(device)
    criterion_cls = nn.BCELoss()
    criterion_reg = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (points, obj_labels, bbox_targets) in enumerate(dataloader):
            points = points.to(device)
            obj_labels = obj_labels.to(device)
            bbox_targets = bbox_targets.to(device)

            optimizer.zero_grad()
            obj_scores, bbox_preds = model(points)
            loss_cls = criterion_cls(obj_scores, obj_labels)
            positive_mask = obj_labels > 0
            if positive_mask.sum() > 0:
                loss_reg = criterion_reg(bbox_preds[positive_mask], bbox_targets[positive_mask])
            else:
                loss_reg = torch.tensor(0.0).to(device)
            loss = loss_cls + loss_reg
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {running_loss / len(dataloader):.4f}')

    model_path = 'pointnet_detector.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as '{model_path}'")

if __name__ == "__main__":
    # Example: fix your paths to forward slash on Linux
    pcd_file_paths = [
        "/home/dalek/Downloads/Issac-Point-Net-main/PCD_Train_Data/labeled_initial_scene1.pcd",
        "/home/dalek/Downloads/Issac-Point-Net-main/PCD_Train_Data/labeled_initial_scene2.pcd",
        "/home/dalek/Downloads/Issac-Point-Net-main/PCD_Train_Data/labeled_initial_scene3.pcd",
        "/home/dalek/Downloads/Issac-Point-Net-main/PCD_Train_Data/labeled_initial_scene4.pcd",
        "/home/dalek/Downloads/Issac-Point-Net-main/PCD_Train_Data/labeled_initial_scene5.pcd",
    ]
    train_pointnet_detector(pcd_file_paths, epochs=500, batch_size=1, lr=0.0001)
