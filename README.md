# HGR-Net
Exploring Hierarchical Graph Representation for Large-Scale Zero-Shot Image Classification

## Requirements
python=3.7.9

pytorch=1.7.1

scipy

scikit-learn

numpy

nltk

## Data
**Step 1** Run ./data/hierarhical.py to reconstruct the hierarchical structure.

**Step 2** Run ./data/remove_irrelevant.py to remove classes that don't fit in the hierarchical structure.

**Step 3** Prepare images and run ./data/train_test_split_backup.py to obtain instance-wise split.

./data/train_test_split.py for low-shot

./data/imagenet21kp.py for ImageNet-21K-P split.

## Train

Copy below command to your scripts

zero-shot classification

```
python main.py --arch RN50 --training_method OM --weights adaptive --training_method OM --sample_strategy topk --lr 3e-7 --w_lr 1e-4 \
--out_ratio 0.25 --in_ratio 0.5 --data_train train --data_test rest --data_split_train train --data_split_test val --batch_size 256
```

low-shot classification

```
python main.py --arch RN50 --training_method OM --weights adaptive --training_method OM --sample_strategy topk --lr 1e-6 --w_lr 1e-4
--out_ratio 0.25 --in_ratio 0.5 --data_train train --data_test rest --data_split_train ls_train --data_split_test ls_val --batch_size 256 --k_shots $k_shots --fetch --fetch_path $zsl_model_path
```

## Test

```
python main.py --train False --load --load_path $model_path --data_train train --data_test rest --data_split_train train --data_split_test zsl_test
```

## Citation
