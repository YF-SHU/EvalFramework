# Entity Alignment via Graph Neural Networks: A Component-level Study

## Data

Two datasets were used: DBP15K and SRPRS. Please unzip data.zip and download the following fasttext vectors and place them under "./data": 
* wiki.en.vec: https://fasttext.cc/docs/en/pretrained-vectors.html
* wiki.{lang}.align.vec: https://fasttext.cc/docs/en/aligned-vectors.html

## Code

### Dependencies

* Python 3.8
* Tensorflow 2.10
* scikit-learn 1.2.0

### Run

* To generate fasttext name embeddings:
```
python preprocess.py
```
* To run a test, e.g., GCN on DBP15K(zh_en) with highway gates and entity name initialisation:
```
python main.py --input data/DBP15K/zh_en/ --embedding_model GCN --skip_conn highway --ent_name_init True
```

## Acknowledgement

Some codes were adapted from OpenEA (https://github.com/nju-websoft/OpenEA/).
