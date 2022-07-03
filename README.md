# HGR-Net
This is the official code repository for ECCV 2022 paper: [Exploring Hierarchical Graph Representation for Large-Scale Zero-Shot Image Classification](https://arxiv.org/abs/2203.01386)

## Requirements
- python=3.7.9
- pytorch=1.7.1
- scipy
- scikit-learn
- numpy
- nltk

## Data

Prepare ImageNet-21K images and organize them like:

```
--wnid1
  --image_name1
  --image_name2
--wnid2
  --image_name1
  --image_name2
```

Preprocess

```
# step 1: reconstruct the hierarchical structure
python ./data/hierarhical.py

# step 2: remove classes that don't fit in the hierarchical structure
python ./data/remove_irrelevant.py

# step 3: instance-wise split
python ./data/train_test_split_backup.py

# ./data/train_test_split.py for low-shot
# ./data/imagenet21kp.py for ImageNet-21K-P split
```

## Train

Copy below command to your scripts

zero-shot classification

```
python main.py --arch RN50 --training_method OM --weights adaptive --training_method OM --sample_strategy topk --lr 3e-7 --w_lr 1e-4 \
--out_ratio 0.25 --in_ratio 0.5 --data_train train --data_test rest --data_split_train train --data_split_test val --batch_size 256

# replace data_split_train and data_split_test with other split
```

low-shot classification

```
python main.py --arch RN50 --training_method OM --weights adaptive --training_method OM --sample_strategy topk --lr 1e-6 --w_lr 1e-4 \
--out_ratio 0.25 --in_ratio 0.5 --data_train train --data_test rest --data_split_train ls_train --data_split_test ls_val --batch_size 256 --k_shots $k_shots --fetch --fetch_path $zsl_model_path
```

## Test

```
python main.py --train False --load --load_path $model_path --data_train train --data_test rest --data_split_train train --data_split_test zsl_test --test_batch_size 512
```

## Citation

  @article{yi2022exploring,
    title={Exploring hierarchical graph representation for large-scale zero-shot image classification},
    author={Yi, Kai and Shen, Xiaoqian and Gou, Yunhao and Elhoseiny, Mohamed},
    journal={ECCV},
    year={2022}
  }
