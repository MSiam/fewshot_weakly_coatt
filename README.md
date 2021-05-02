# Weakly Supervised Few-shot Object Segmentation using Co-Attention with Visual and Semantic Embeddings

Training
```
python main.py -train 1 -model_type coatt_nwe -dataset_name pascal -fold FOLD -ckpt CHECKPOINTDIR
```

Testing 
```
python main.py -train 0 -test_multi_runs 1 -model_type coatt_nwe -dataset_name pascal -fold FOLD -ckpt CHECKPOINT_DIR
```
