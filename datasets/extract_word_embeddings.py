import gensim
import numpy as np
#wget -P /root/input/ -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

word2vec = gensim.models.KeyedVectors.load_word2vec_format('word2vec_weights/GoogleNews-vectors-negative300.bin.gz', binary=True)
classes = ['plane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'table', 'dog', 'horse',
            'motorbike', 'person', 'plant',
            'sheep', 'sofa', 'train', 'monitor']

embeddings = {}
for cls in classes:
    embeddings[cls] = word2vec[cls]

embeddings_stored = np.load('embeddings.npy', allow_pickle=True).item()
error = 0
for cls in classes:
    error = np.sum(embeddings[cls]-embeddings_stored[cls])
    if error !=0:
        print('error in ', cls)
    else:
        print('passed ', cls)
