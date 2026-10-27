import os
from monai.transforms import LoadImage

loader = LoadImage(image_only=True)

path = os.path.join(
    "..",
    "Task07_Pancreas",
    "imagesTr",
    "pancreas_001.nii"
)

img = loader(path)

print(img.shape)