In this work, I build the Translation Machine from English to French by Transformer described in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

For the data, I use the parallel corpus French-English from [European Parliament dataset version 7](https://www.statmt.org/europarl/).

For its parameters, I set:
 + MAX_LENGTH = 40
 + BATCH_SIZE = 64
 + BUFFER_SIZE = 20000
 + num_layers = 4
 + d_model = 216
 + dff = 512 
 + num_heads = 8 
 
I ran 10 epochs on Colab Pro (GPU T4) and it took 23 mins each epoch.
