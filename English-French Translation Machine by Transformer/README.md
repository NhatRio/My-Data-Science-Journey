In this work, I build the Translation Machine from English to French by Transformer in the paper ["All Is All You Need"](https://arxiv.org/abs/1706.03762)
For the data, I use the parallel corpus French-English from [European Parliament dataset version 7](https://www.statmt.org/europarl/).

For its parameters, I set:
 + BATCH_SIZE = 64
 + BUFFER_SIZE = 20000
 + num_layers = 4
 + d_model = 216
 + dff = 512 
 + num_heads = 8 
 
I ran 10 epochs and it took 23 mins/epoch.
