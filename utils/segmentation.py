import os
import torch
import numpy as np

from monai.networks.nets import UNet

from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Spacing,
    ScaleIntensity,
    Resize,
    EnsureType
)

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# MODEL
# =====================================================

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)

# =====================================================
# LOAD TRAINED MODEL
# =====================================================

model.load_state_dict(
    torch.load(
        "models/pancreas_unet.pth",
        map_location=device
    )
)

model.eval()

# =====================================================
# PREPROCESSING TRANSFORMS
# =====================================================

transforms = Compose([

    LoadImage(image_only=True),

    EnsureChannelFirst(),

    Orientation(axcodes="RAS"),

    Spacing(
        pixdim=(1.5, 1.5, 2.0),
        mode="bilinear"
    ),

    ScaleIntensity(),

    Resize((128, 128, 128)),

    EnsureType()

])

# =====================================================
# SEGMENTATION FUNCTION
# =====================================================

def segment_pancreas(image_path):

    print("=" * 60)
    print("STEP 1: Loading image")
    print("Image Path :", image_path)
    print("Exists     :", os.path.exists(image_path))

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("File Size  :", os.path.getsize(image_path), "bytes")
    print("=" * 60)

    try:

        image = transforms(image_path)

    except Exception as e:

        print("Error while loading image:")
        print(repr(e))
        raise

    print("STEP 2: Image loaded successfully")
    print("Image Shape :", image.shape)

    input_tensor = image.unsqueeze(0).to(device)

    print("STEP 3: Input Tensor Shape :", input_tensor.shape)

    with torch.no_grad():

        print("STEP 4: Running segmentation model...")

        output = model(input_tensor)

    print("STEP 5: Segmentation completed")

    mask = torch.argmax(output, dim=1)

    image = image.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()

    print("STEP 6: Returning segmentation results")
    print("=" * 60)

    return image, mask