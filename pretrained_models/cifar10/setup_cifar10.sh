# Download trained GANs from places
if [ ! -d "data/cifar10_gen" ]; then
    mkdir data/cifar10_gen
fi
rm -f data/cifar10_gen/*
wget https://s3.amazonaws.com/robust-manifold-defense/cifar10_gen.zip -O data/cifar10_gen/data.zip
chmod 775 data/cifar10_gen/data.zip
unzip data/cifar10_gen/data.zip -d data/cifar10_gen
rm -f data/cifar10_gen/data.zip

# Download trained models from places
if [ ! -d "data/cifar10_classifier" ]; then
    mkdir data/cifar10_classifier
fi
rm -f data/cifar10_classifier/*
wget https://s3.amazonaws.com/robust-manifold-defense/cifar10_classifier.zip -O data/cifar10_classifier/data.zip
chmod 775 data/cifar10_classifier/data.zip
unzip data/cifar10_classifier/data.zip -d data/cifar10_classifier
rm -f data/cifar10_classifier/data.zip

# Download training data from places
if [ ! -d "data/cifar10_train" ]; then
    mkdir data/cifar10_train
fi
rm -f data/cifar10_train/*
wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O data/cifar10_train/data.tar.gz
chmod 775 data/cifar10_train/data.tar.gz
tar -xvzf data/cifar10_train/data.tar.gz -C data/cifar10_train
rm -f data/cifar10_train/data.tar.gz
