import os

dataset_path = r"D:\himanshu\2d\classified_dataset"

print("\n========== 2D DATASET STRUCTURE ==========\n")

total_images = 0

for class_name in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):

        files = os.listdir(class_path)

        image_count = 0
        file_types = set()

        for file in files:

            ext = os.path.splitext(file)[1].lower()

            if ext in [".jpg", ".jpeg", ".png"]:
                image_count += 1

            file_types.add(ext)

        total_images += image_count

        print(f"Class: {class_name}")
        print("Images:", image_count)
        print("File types:", file_types)
        print("--------------------------------")

print("\nTOTAL IMAGES IN DATASET:", total_images)