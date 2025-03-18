from src.data.image_only_dataset import ImageOnlyDataset
from src.data.util.dataset import download_and_unzip
from torchvision.datasets import ImageFolder
from torchvision import transforms


def load_anime_faces_dataset(folder, image_size):
    path = download_and_unzip("splcher", "animefacedataset", folder)

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    return ImageOnlyDataset(ImageFolder(path, transform=transform))
