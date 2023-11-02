# Entity Alignment via Graph Neural Networks: A Component-level Study

## Data

Two datasets were used: DBP15K and SRPRS. 

## Code

### Dependencies

* Python 3.8
* Tensorflow 2.10
* scikit-learn 1.2.0


### Example run

To run GCN on DBP15K(zh_en) with highway gates and entity name initialisation:
```
python main.py --input data/DBP15K/zh_en/ --embedding_model GCN --skip_conn highway --ent_name_init True
```

## Acknowledgement

Some codes were adapted from OpenEA (https://github.com/nju-websoft/OpenEA/).
