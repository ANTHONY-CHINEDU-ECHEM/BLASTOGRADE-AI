# Training on real blastocyst images

BlastoGrade AI reads any folder that follows this layout:

```
my_dataset/
    labels.csv
    images/
        embryo_0001.png
        ...
```

`labels.csv` needs five columns.

| column    | content                                                        |
|-----------|----------------------------------------------------------------|
| image     | path relative to the dataset folder                            |
| expansion | Gardner expansion grade, 1 to 6                                |
| icm       | inner cell mass grade A, B or C (blank when expansion is 1 or 2) |
| te        | trophectoderm grade A, B or C (blank when expansion is 1 or 2)  |
| split     | train, val or test                                             |

Split by patient, not by image, so that sibling embryos never straddle the train and test sets.

Then point `data.root` in `configs/config.yaml` at the folder, set `model.pretrained: true` to start from
ImageNet weights, raise `training.image_size` to 224 and run `make train` followed by `make evaluate`.

Public sources with Gardner style annotations include the annotated human blastocyst dataset published in
Scientific Data (Kromp and colleagues, 2023) and the embryo image sets on Kaggle. The STORK dataset provides
good versus poor quality labels rather than full Gardner grades; it can be used by training the expansion head
only and leaving `icm` and `te` blank. Check each licence before use. The Grad CAM pointing game in the
evaluation step needs the `icm_x`, `icm_y` and `icm_r` columns and is skipped when they are absent.
