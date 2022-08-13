import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_data(batch_size):
    ##Transformations
    train_transforms = transforms.Compose(
        [  # Compose makes it possible to have many transforms
            #transforms.ColorJitter(brightness=0.5),
            #transforms.RandomRotation(degrees=45),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomVerticalFlip(p=0.05),
            transforms.ToTensor(),
            #transforms.Normalize( mean=[0.1307], std=[0.5]),
        ]
    )

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.1307],std=[0.3081]),
    ])

    #Load the Data
    train_dataset = datasets.MNIST(root="C:/Users/rajar/PycharmProjects/CNN Architectures/dataset/", train=True, transform=train_transforms, download=True)
    test_dataset = datasets.MNIST(root="C:/Users/rajar/PycharmProjects/CNN Architectures/dataset/", train=False, transform=test_transforms, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader