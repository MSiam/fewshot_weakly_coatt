# Weakly Supervised Few-shot Object Segmentation using Co-Attention with Visual and Semantic Embeddings

Official Implementation for our [IJCAI 2020 Paper](https://www.ijcai.org/Proceedings/2020/0120.pdf).

Training
```
python main.py -train 1 -model_type coatt_nwe -dataset_name pascal -fold FOLD -ckpt CHECKPOINTDIR
```

Testing 
```
python main.py -train 0 -test_multi_runs 1 -model_type coatt_nwe -dataset_name pascal -fold FOLD -ckpt CHECKPOINT_DIR
```
