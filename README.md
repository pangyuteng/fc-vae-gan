

```


docker run -u $(id -u):$(id -g) --runtime=nvidia -it -v /mnt:/mnt -w $PWD fcvae bash


# test data-generator
python data_gen.py

# test model
python models.py

# start training
python train.py ped-ct-seg.csv

# TODO: eval
get latent variables, dim reduction with tsne, overlay organ classification.


```

https://groups.csail.mit.edu/vision/datasets/ADE20K/

https://arxiv.org/abs/1804.04488


