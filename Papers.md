# 1. Variational Graph Auto-Encoders 
## Kipf, Welling (2016) [paper](https://arxiv.org/pdf/1611.07308.pdf)
For link prediction in citation networks. 
It makes use of latent variables and learns latent rerpresentations for an undirected graph. 
Undirected graphs with GCN encoder + inner product encoder. It incorporates node features. 

# 2. Variational Auto-Encoders
* [tutorial](https://arxiv.org/pdf/1606.05908.pdf)
* [tutorial2](https://towardsdatascience.com/tutorial-on-variational-graph-auto-encoders-da9333281129)
* [tutorial3](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)
* Main idea: The main idea of a variational autoencoder is that it embeds the input X to a distribution rather than a point. And then a random sample Z is taken from the distribution rather than generated from encoder directly.
