# Download trained GANs from places
if [ ! -d "data/CelebA_gen" ]; then
    mkdir data/CelebA_gen
fi
rm -f data/CelebA_gen/*
wget https://s3.amazonaws.com/robust-manifold-defense/celebA_gen.zip -O data/CelebA_gen/data.zip
chmod 775 data/CelebA_gen/data.zip
unzip data/CelebA_gen/data.zip -d data/CelebA_gen
rm -f data/CelebA_gen/data.zip

# Download trained models from places
if [ ! -d "data/CelebA_classifier" ]; then
    mkdir data/CelebA_classifier
fi
rm -f data/CelebA_classifier/*
wget https://s3.amazonaws.com/robust-manifold-defense/celebA_classifier.zip -O data/CelebA_classifier/data.zip
chmod 775 data/CelebA_classifier/data.zip
unzip data/CelebA_classifier/data.zip -d data/CelebA_classifier
rm -f data/CelebA_classifier/data.zip

# Download training data from places
if [ ! -d "data/CelebA_train" ]; then
    mkdir data/CelebA_train
fi
rm -f data/CelebA_train/*
wget https://s3.amazonaws.com/robust-manifold-defense/celebA_train.zip -O data/CelebA_train/data.zip
chmod 775 data/CelebA_train/data.zip
unzip data/CelebA_train/data.zip -d data/CelebA_train
rm -f data/CelebA_train/data.zip
