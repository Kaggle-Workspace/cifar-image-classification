mkdir -p "data"
cp -r "/kaggle/input/cifar-10" "/root/cifar-image-classification/data/"

7z x ./data/train.7z -o"/root/cifar-image-classification/data/"
7z x ./data/test.7z -o"/root/cifar-image-classification/data/"
