

```

docker build -t fcvae .
docker run -u $(id -u):$(id -g) --runtime=nvidia -it -v /mnt:/mnt -w $PWD fcvae bash


# test data-generator
python data_gen.py

# test model
python models.py

# start training
python train.py brats19.csv

# monitor tasks
docker run -u $(id -u):$(id -g) -p 6006:6006 -it -v /mnt:/mnt -w $PWD fcvae bash
tensorboard --bind_all --logdir=log



# visualize latent variables via tsne
python clustering.py /mnt/hd2/data/brats2019/MICCAI_BraTS_2019_Data_Training/LGG/BraTS19_TCIA13_653_1 cluster_tmp

# segment using knn
python inference.py /mnt/hd2/data/brats2019/MICCAI_BraTS_2019_Data_Training/LGG/BraTS19_TCIA13_653_1

# TODO: eval , compute "gan" metrics...


notes

+ training is somewhat stable with kernel at 1,7,7



```

https://groups.csail.mit.edu/vision/datasets/ADE20K/

https://arxiv.org/abs/1804.04488

https://www.youtube.com/watch?v=oHtqlRIsXcQ
https://keras.io/examples/generative/vq_vae
https://lilianweng.github.io/posts/2018-08-12-vae

https://nips.cc/media/neurips-2021/Slides/21895.pdf
https://www.youtube.com/watch?v=7l6fttRJzeU



https://arxiv.org/pdf/1804.04488.pdf
https://www.nature.com/articles/s41598-021-84547-5

