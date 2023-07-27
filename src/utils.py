"""
What's inside:
1. reorganize folder structure
2. pred_and_plot_image
"""
import os
import random
import shutil
from typing import List, Tuple
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

def reorganize_folder_structure(
        source_folder,
        destination_folder,
        train_ratio,
        valid_ratio,
        random_seed
        ):
    """
    reorg the folder to meet the pytorch dataloader criteria
    """
    # Create the destination folders
    train_folder = destination_folder / "train"
    valid_folder = destination_folder / "valid"
    test_folder = destination_folder / "test"
    os.makedirs(train_folder, exist_ok=False)
    os.makedirs(valid_folder, exist_ok=False)
    os.makedirs(test_folder, exist_ok=False)

    # Loop through the source folder
    for class_folder in os.listdir(source_folder):
        if os.path.isdir(os.path.join(source_folder, class_folder)):
            # Create the destination class folders
            train_class_folder = train_folder / class_folder
            valid_class_folder = valid_folder / class_folder
            test_class_folder = test_folder / class_folder
            os.makedirs(train_class_folder, exist_ok=False)
            os.makedirs(valid_class_folder, exist_ok=False)
            os.makedirs(test_class_folder, exist_ok=False)

            # Get the list of image files in the class folder
            image_files = [
                f for f in os.listdir(os.path.join(source_folder, class_folder))
                if os.path.isfile(os.path.join(source_folder, class_folder, f))
            ]

            # Shuffle the image files
            random.Random(random_seed).shuffle(image_files)

            # Split the files into train, valid, and test sets
            train_count = int(train_ratio * len(image_files))
            valid_count = int(valid_ratio * len(image_files))
            train_files = image_files[:train_count]
            valid_files = image_files[train_count:train_count+valid_count]
            test_files = image_files[train_count+valid_count:]

            # Move the files to the corresponding folders
            for file in train_files:
                src_path = os.path.join(source_folder, class_folder, file)
                dst_path = os.path.join(train_class_folder, file)
                shutil.copy(src_path, dst_path)

            for file in valid_files:
                src_path = os.path.join(source_folder, class_folder, file)
                dst_path = os.path.join(valid_class_folder, file)
                shutil.copy(src_path, dst_path)

            for file in test_files:
                src_path = os.path.join(source_folder, class_folder, file)
                dst_path = os.path.join(test_class_folder, file)
                shutil.copy(src_path, dst_path)

def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str, 
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: transforms = None,
                        device: torch.device=device
                        ):
    
    
    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ### 

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability 
    # plt.figure()
    plt.imshow(img)
    plt.title(
        f"""
        Pred: {class_names[target_image_pred_label]} 
        Prob: {target_image_pred_probs.max():.3f}
        """
    )
    plt.axis(False)