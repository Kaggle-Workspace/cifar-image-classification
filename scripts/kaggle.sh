mkdir -p "/root/cifar-image-classification/data/"
cp -r "/kaggle/input/cifar-10" "/root/cifar-image-classification/data/"

7z x /root/cifar-image-classification/data/train.7z -o"/root/cifar-image-classification/data/"
7z x /root/cifar-image-classification/data/test.7z -o"/root/cifar-image-classification/data/"
