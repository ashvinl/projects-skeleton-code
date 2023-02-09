import torch


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset .
    """

    def __init__(self, items, labels):
        self.items = items
        self.labels = labels

    def __getitem__(self, index):
        item = self.items[index]
        label = self.labels[index]

        return item, label

    def __len__(self):
        return len(self.labels)
