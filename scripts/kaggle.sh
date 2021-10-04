rm -rf "/root/cifar-image-classification/data"
mkdir -p "/root/cifar-image-classification/data/"
cp -r "/kaggle/input/cifar-10/" "/root/cifar-image-classification/data/"

7z x /root/cifar-image-classification/data/cifar-10/train.7z -o"/root/cifar-image-classification/data/cifar-10"
7z x /root/cifar-image-classification/data/cifar-10/test.7z -o"/root/cifar-image-classification/data/cifar-10"
