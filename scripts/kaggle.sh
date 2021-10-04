rm -rf "/root/cifar-image-classification/input"
mkdir -p "/root/cifar-image-classification/input/"
cp -r "/kaggle/input/cifar-10/" "/root/cifar-image-classification/input/"

7z x /root/cifar-image-classification/input/cifar-10/train.7z -o"/root/cifar-image-classification/input/cifar-10"
7z x /root/cifar-image-classification/input/cifar-10/test.7z -o"/root/cifar-image-classification/input/cifar-10"
