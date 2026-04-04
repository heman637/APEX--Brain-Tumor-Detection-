import os

root_path = r"D:\himanshu"

print("\n========== ROOT DIRECTORY ==========\n")

for item in os.listdir(root_path):
    print(item)


# ------------------------------------------------
# 2D DATASET FULL CHECK
# ------------------------------------------------

path_2d = os.path.join(root_path, "2d")

print("\n========== 2D DATASET FULL STRUCTURE ==========\n")

total_images = 0

for root, dirs, files in os.walk(path_2d):

    print("\nFolder:", root)

    image_count = 0
    file_types = set()

    for file in files:

        ext = os.path.splitext(file)[1].lower()

        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            image_count += 1
            total_images += 1

        file_types.add(ext)

    print("Images in folder:", image_count)

    if file_types:
        print("File types:", file_types)

print("\nTOTAL 2D IMAGES FOUND:", total_images)


# ------------------------------------------------
# 3D DATASET FULL CHECK
# ------------------------------------------------

path_3d = os.path.join(root_path, "3d")

print("\n========== 3D DATASET FULL STRUCTURE ==========\n")

total_nii = 0

for root, dirs, files in os.walk(path_3d):

    print("\nFolder:", root)

    nii_count = 0

    for file in files:

        if file.endswith(".nii") or file.endswith(".nii.gz"):
            nii_count += 1
            total_nii += 1

    print("NIfTI MRI files:", nii_count)

print("\nTOTAL MRI FILES FOUND:", total_nii)