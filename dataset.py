"""
    RIS-GAN
    func  : ISTD数据集
    Author: Chen Yu
    Date  : 2020.10.20
"""
import torch
from torch.utils.data import Dataset

from utils import match_samples_path, load_image_pair, get_negative_residual_image, get_inverse_illuminate_map


class IstdDataset(Dataset):

    def __init__(self, data_dir):
        self.samples_path = match_samples_path(data_dir)

    def __getitem__(self, idx):
        sample_path = self.samples_path[idx]
        shadow_data, shadow_removal_data = load_image_pair(sample_path)
        negative_residual_image = get_negative_residual_image(shadow_data, shadow_removal_data)
        inverse_illuminate_map = get_inverse_illuminate_map(shadow_data, shadow_removal_data)

        shadow_data = torch.tensor(shadow_data, dtype=torch.float32).transpose(2, 0)
        shadow_removal_data = torch.tensor(shadow_removal_data, dtype=torch.float32).transpose(2, 0)
        negative_residual_image = torch.tensor(negative_residual_image, dtype=torch.float32).transpose(2, 0)
        inverse_illuminate_map = torch.tensor(inverse_illuminate_map, dtype=torch.float32).transpose(2, 0)
        return shadow_data, shadow_removal_data, negative_residual_image, inverse_illuminate_map

    def __len__(self):
        return len(self.samples_path)


if __name__ == '__main__':
    data_dir = 'ISTD_Dataset/train'
    dataset = IstdDataset(data_dir)
    it = iter(dataset)
    sample = next(it)
    print(sample[0].shape)
    print(sample[1].shape)
    print(sample[2].shape)
    print(sample[3].shape)
