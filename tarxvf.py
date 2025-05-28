# Author: Arlse
# tar code for ILSVRC2012_img_train.tar

import os
import tarfile


def tarxvf(tar_file, target_dir):
    """
    Extract the tar file to the target directory.
    :param tar_file: str, the tar file path.
    :return: None
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with tarfile.open(tar_file, 'r') as tar:
        members = tar.getmembers()
        for member in members:
            if member.name.endswith('.tar'):
                subpath = os.path.join(target_dir, member.name.split('.')[0])
                if not os.path.exists(subpath):
                    os.makedirs(subpath)
                with tar.extractfile(member) as subtar:
                    subtar = tarfile.open(fileobj=subtar)
                    subtar.extractall(subpath)



if __name__ == "__main__":
    tar_file = "/datasets/Imagenet2012/ILSVRC2012_img_train.tar"
    target_dir = "/datasets/Imagenet2012/train"
    tarxvf(tar_file, target_dir)
    print("Extracted successfully!")