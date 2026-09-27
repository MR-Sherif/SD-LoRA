# Dataset preparation

Put all datasets beneath one parent directory and pass that parent as `--root_path` to `main.py`. The folder names below follow the loaders in this repository. The [Tip-Adapter dataset guide](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/DATASET.md) and the author's [JPTA dataset guide](https://github.com/MR-Sherif/JPTA/blob/main/DATASETS.md) provide download and split-file links.

| Dataset config | Folder under the data root | Key contents expected by the loader |
| --- | --- | --- |
| `imagenet` | `imagenet/` | `images/train/` and `images/val/` with ImageNet class folders |
| `caltech101` | `caltech-101/` | `101_ObjectCategories/`, `split_zhou_Caltech101.json` |
| `oxford_pets` | `oxford_pets/` | `images/`, `split_zhou_OxfordPets.json` |
| `stanford_cars` | `cars/` | `cars_train/`, `cars_test/`, annotations, `split_zhou_StanfordCars.json` |
| `oxford_flowers` | `flowers/` | `jpg/`, `imagelabels.mat`, `cat_to_name.json`, `split_zhou_OxfordFlowers.json` |
| `food101` | `food-101/` | `images/`, `split_zhou_Food101.json` |
| `aircrafts` | `aircrafts/` | `images/`, `variants.txt`, `images_variant_train.txt`, `images_variant_val.txt`, `images_variant_test.txt` |
| `sun397` | `sun397/` | `SUN397/`, `split_zhou_SUN397.json` |
| `dtd` | `dtd/` | `images/`, `split_zhou_DescribableTextures.json` |
| `eurosat` | `eurosat/` | `2750/`, `split_zhou_EuroSAT.json` |
| `ucf101` | `ucf101/` | `UCF-101-midframes/`, `split_zhou_UCF101.json` |

Download datasets from their original providers and obtain the split JSON files from the linked dataset guides. Dataset files are not part of this repository. Existing datasets can be linked into the expected folder names instead of copied.

The original ImageNet loader uses `images/val/` as both validation and test data. This behavior is preserved in the released source; take it into account when interpreting ImageNet metrics from the historical scripts.
