echo 'I was not able to get this to find the remote files. I guess the files are missing from the s3 bucket. But who cares, its just mnist'

# Download trained GANs from places
if [ ! -d "data/MNIST_gen" ]; then
    mkdir data/MNIST_gen
fi
rm -f data/MNIST_gen/*
wget https://s3.amazonaws.com/robust-manifold-defense/mnist_gen.zip -O data/MNIST_gen/data.zip
chmod 775 data/MNIST_gen/data.zip
unzip data/MNIST_gen/data.zip -d data/MNIST_gen
rm -f data/MNIST_gen/data.zip

# Download trained models from places
if [ ! -d "data/MNIST_classifier" ]; then
    mkdir data/MNIST_classifier
fi
rm -f data/MNIST_classifier/*
wget https://s3.amazonaws.com/robust-manifold-defense/mnist_classifier.zip -O data/MNIST_classifier/data.zip
chmod 775 data/MNIST_classifier/data.zip
unzip data/MNIST_classifier/data.zip -d data/MNIST_classifier
rm -f data/MNIST_classifier/data.zip

# Download training data from places
if [ ! -d "data/MNIST_train" ]; then
    mkdir data/MNIST_train
fi
rm -f data/MNIST_train/*
wget https://s3.amazonaws.com/robust-manifold-defense/mnist_train.zip -O data/MNIST_train/data.zip
chmod 775 data/MNIST_train/data.zip
unzip data/MNIST_train/data.zip -d data/MNIST_train
rm -f data/MNIST_train/data.zip
