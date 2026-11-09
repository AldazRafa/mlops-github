import os
import argparse
import logging
from glob import glob
import math
import random

def main():
    """Main function of the script."""

    SEED = 42

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", help="All the datasets to combine")
    parser.add_argument("--training_data_output", type=str, help="path to training output data")
    parser.add_argument("--testing_data_output", type=str, help="path to testing output data")
    parser.add_argument("--validation_data_output", type=str, default=None, help="optional path for validation images")
    parser.add_argument("--split_size", type=int, help="Percentage to use as Testing data")
    parser.add_argument("--validation_size", type=int, default=20, help="Percentage of remaining images for validation")
    args = parser.parse_args()

    if not 0 < args.split_size < 100:
        parser.error("--split_size must be between 1 and 99")

    output_dirs = [args.training_data_output, args.testing_data_output]
    if args.validation_data_output:
        if not 0 < args.validation_size < 100:
            parser.error("--validation_size must be between 1 and 99")
        output_dirs.append(args.validation_data_output)
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)
        for old_image in glob(os.path.join(output_dir, "*.jpg")):
            os.remove(old_image)

    print("input data:", args.datasets)
    print("Training folder:", args.training_data_output)
    print("Testing folder:", args.testing_data_output)
    print("Validation folder:", args.validation_data_output)
    print("Split size:", args.split_size)

    train_test_split_factor = args.split_size / 100 # Alias
    datasets = args.datasets

    for dataset in datasets:
        animal_images = glob(dataset + "/*.jpg")
        if not animal_images:
            raise FileNotFoundError(f"No .jpg images found in dataset: {dataset}")
        print(f"Found {len(animal_images)} images for {dataset}")

        ## Concatenate the names for the animal_name and the img_path. Don't put a / between, because the img_path already contains that
        ## animal_images = [(default_datastore, f'processed_animals/{animal_name}{img_path}') for img_path in animal_images] # Make sure the paths are actual DataPaths
    
        random.seed(SEED) # Use the same random seed as I use and defined in the earlier cells
        random.shuffle(animal_images) # Shuffle the data so it's randomized

        ## Testing images
        amount_of_test_images = math.ceil(len(animal_images) * train_test_split_factor) # Get a small percentage of testing images

        animal_test_images = animal_images[:amount_of_test_images]
        remaining_images = animal_images[amount_of_test_images:]
        if args.validation_data_output:
            amount_of_validation_images = math.ceil(
                len(remaining_images) * args.validation_size / 100
            )
            animal_validation_images = remaining_images[:amount_of_validation_images]
            animal_training_images = remaining_images[amount_of_validation_images:]
        else:
            animal_validation_images = []
            animal_training_images = remaining_images

        # Add them all to the other ones
        print(animal_test_images[:5])

        # Write the data to the output
        for img in animal_test_images:
            # Open the img, which is a string filepath, then save it to the args.testing_data_output directory
            with open(img, "rb") as f:
                with open(os.path.join(args.testing_data_output, os.path.basename(img)), "wb") as f2:
                    f2.write(f.read())

        if args.validation_data_output:
            for img in animal_validation_images:
                with open(img, "rb") as f:
                    with open(os.path.join(args.validation_data_output, os.path.basename(img)), "wb") as f2:
                        f2.write(f.read())

        for img in animal_training_images:
            # Open the image and copy it to the training output directory.
            with open(img, "rb") as f:
                with open(os.path.join(args.training_data_output, os.path.basename(img)), "wb") as f2:
                    f2.write(f.read())

if __name__ == "__main__":
    main()
