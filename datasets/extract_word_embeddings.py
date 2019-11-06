import gensim
import numpy as np
import sys
from gensim.models.wrappers import FastText

def extract_embed(model, cls):
    if cls not in model:
        cls_splits = cls.split('_')
        embeddings = None
        for split in cls_splits:
            if embeddings is None:
                embeddings = np.array(model[split])
            else:
                embeddings += np.array(model[split])
    else:
        embeddings = model[cls]
    return embeddings

def main():
    if sys.argv[2] in ['fasttext', 'concat']:
        #wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.bin.zip
        model_fst = FastText.load_fasttext_format('wiki-news-300d-1M-subword', encoding='utf-8')
    else:
        model_fst = None

    if sys.argv[2] in ['word2vec', 'concat']:
        #wget -P /root/input/ -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
        model_w2vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    else:
        model_w2vec = None

    if sys.argv[1] == 'pascal':
        classes = ['plane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'table', 'dog', 'horse',
                    'motorbike', 'person', 'plant',
                    'sheep', 'sofa', 'train', 'monitor']
    else:
        classes_f = open('datasets/coco_classes.txt', 'r')
        classes = []
        for line in classes_f:
            classes.append(line.strip().replace(' ', '_'))
        classes_f.close()


    embeddings = {}
    for cls in classes:
        if model_w2vec is not None:
            embeddings_w2vec = extract_embed(model_w2vec, cls)
        else:
            embeddings_w2vec = None

        if model_fst is not None:
            embeddings_fst = extract_embed(model_fst, cls)
        else:
            embeddings_fst = None

        if embeddings_w2vec is not None and embeddings_fst is not None:
            embedding = np.concatenate((embeddings_w2vec, embeddings_fst), axis=0)
        elif embeddings_w2vec is not None:
            embedding = embeddings_w2vec
        else:
            embedding = embeddings_fst

        embeddings[cls] = embedding

    if len(sys.argv) > 3: # Confirm embeddings
        embeddings_stored = np.load('embeddings_%s_%s.npy'%(sys.argv[2], sys.argv[1]), allow_pickle=True).item()
        error = 0
        for cls in classes:
            error = np.sum(embeddings[cls]-embeddings_stored[cls])
            if error !=0:
                print('error in ', cls)
            else:
                print('passed ', cls)
    else: # save embeddings
        np.save('embeddings_%s_%s.npy'%(sys.argv[2], sys.argv[1]), embeddings, allow_pickle=True)

main()
