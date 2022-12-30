import torch
from torch.utils.data import Dataset,DataLoader

class TabularDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        # self.class_weights = class_weights

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        features = torch.tensor(row.values, dtype=torch.float)
        # label = torch.tensor(row['label'], dtype=torch.float)
        # weight = torch.tensor(self.class_weights[row['label']], dtype=torch.float)
        return features, label, weight


def get_dataloaders(batch_size, train_df, test_df):
    train_dataset = TabularDataset(train_df)
    test_dataset = TabularDataset(test_df)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader