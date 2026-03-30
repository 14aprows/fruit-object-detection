import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform():
    return A.Compose(
        [
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Blur(blur_limit=3, p=0.1),

            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),

            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"]
        )
    )

def get_valid_transform():
    return A.Compose(
        [
            A.Resize(512, 512),

            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),

            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"]
        )
    )