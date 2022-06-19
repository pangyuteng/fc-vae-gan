

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

https://www.youtube.com/watch?v=oHtqlRIsXcQ
https://keras.io/examples/generative/vq_vae
https://lilianweng.github.io/posts/2018-08-12-vae

https://nips.cc/media/neurips-2021/Slides/21895.pdf
https://www.youtube.com/watch?v=7l6fttRJzeU
