# sar_vessel_detect

Code for the AI2 Skylight team's submission in the xView3 competition (https://iuu.xview.us) for vessel detection in Sentinel-1 SAR images.
See `whitepaper.pdf` for a summary of our approach.

Dependencies
------------

Install dependiences using conda:

```
cd sar_vessel_detect/
conda env create -f environment.yml
```


Pre-processing
--------------

First, ensure that training and validation scenes are extracted to the same directory, e.g. `/xview3/all/images/`.
The training and validation labels should be concatenated and written to a CSV file like `/xview3/all/labels.csv`.

Prior to training, the large scenes must be split up into 800x800 windows (chips).
Set paths and parameters in `data/configs/chipping_config.txt`, and then run:

```
cd sar_vessel_detect/src/
python -m xview3.processing.preprocessing ../data/configs/chipping_config.txt
```


Initial Training
----------------

We first train a model on the 50 xView3-Validation scenes only.
We will apply this model in the xView3-Train scenes, and incorporate high-confidence predictions as additional labels.
This is because xView3-Train scenes are not comprehensively labeled since most labels are derived automatically from AIS tracks.

To train, set paths and parameters in `data/configs/initial.txt`, and then run:

```
python -m xview3.training.train ../data/configs/initial.txt
```

Apply the trained model in xView3-Train, and incorporate high-confidence predictions as additional labels:

```
python -m xview3.infer.inference --image_folder /xview3/all/images/ --weights ../data/models/initial/best.pth --output out.csv --config_path ../data/configs/initial.txt --padding 400 --window_size 3072 --overlap 20 --scene_path ../data/splits/xview-train.txt
python -m xview3.eval.prune --in_path out.csv --out_path out-conf80.csv --conf 0.8
python -m xview3.misc.pred2label out-conf80.csv /xview3/all/chips/ out-conf80-tolabel.csv
python -m xview3.misc.pred2label_concat /xview3/all/chips/chip_annotations.csv out-conf80-tolabel.csv out-conf80-tolabel-concat.csv
python -m xview3.eval.prune --in_path out-conf80-tolabel-concat.csv --out_path out-conf80-tolabel-concat-prune.csv --nms 10
python -m xview3.misc.pred2label_fixlow out-conf80-tolabel-concat-prune.csv
python -m xview3.misc.pred2label_drop out-conf80-tolabel-concat-prune.csv out.csv out-conf80-tolabel-concat-prune-drop.csv
mv out-conf80-tolabel-concat-prune-drop.csv ../data/xval1b-conf80-concat-prune-drop.csv
```


Final Training
--------------

Now we can train the final object detection model.
Set paths and parameters in `data/configs/final.txt`, and then run:

```
python -m xview3.training.train ../data/configs/final.txt
```


Attribute Prediction
--------------------

We use a separate model to predict is_vessel, is_fishing, and vessel length.

```
python -m xview3.postprocess.v2.make_csv /xview3/all/chips/chip_annotations.csv out.csv ../data/splits/our-train.txt /xview3/postprocess/labels.csv
python -m xview3.postprocess.v2.get_boxes /xview3/postprocess/labels.csv /xview3/all/chips/ /xview3/postprocess/boxes/
python -m xview3.postprocess.v2.train /xview3/postprocess/model.pth /xview3/postprocess/labels.csv /xview3/postprocess/boxes/
```


Inference
---------

Suppose that test images are in a directory like `/xview3/test/images/`. First, apply the object detector:

```
python -m xview3.infer.inference --image_folder /xview3/test/images/ --weights ../data/models/final/best.pth --output out.csv --config_path ../data/configs/final.txt --padding 400 --window_size 3072 --overlap 20
python -m xview3.eval.prune --in_path out.csv --out_path out-prune.csv --nms 10
```

Now apply the attribute prediction model:

```
python -m xview3.postprocess.v2.infer /xview3/postprocess/model.pth out-prune.csv /xview3/test/chips/ out-prune-attribute.csv attribute
```


Test-time Augmentation
----------------------

We employ test-time augmentation in our final submission, which we find provides a small 0.5% performance improvement.

```
python -m xview3.infer.inference --image_folder /xview3/test/images/ --weights ../data/models/final/best.pth --output out-1.csv --config_path ../data/configs/final.txt --padding 400 --window_size 3072 --overlap 20
python -m xview3.infer.inference --image_folder /xview3/test/images/ --weights ../data/models/final/best.pth --output out-2.csv --config_path ../data/configs/final.txt --padding 400 --window_size 3072 --overlap 20 --fliplr True
python -m xview3.infer.inference --image_folder /xview3/test/images/ --weights ../data/models/final/best.pth --output out-3.csv --config_path ../data/configs/final.txt --padding 400 --window_size 3072 --overlap 20 --flipud True
python -m xview3.infer.inference --image_folder /xview3/test/images/ --weights ../data/models/final/best.pth --output out-4.csv --config_path ../data/configs/final.txt --padding 400 --window_size 3072 --overlap 20 --fliplr True --flipud True
python -m xview3.eval.ensemble out-1.csv out-2.csv out-3.csv out-4.csv out-tta.csv
python -m xview3.eval.prune --in_path out-tta.csv --out_path out-tta-prune.csv --nms 10
python -m xview3.postprocess.v2.infer /xview3/postprocess/model.pth out-tta-prune.csv /xview3/test/chips/ out-tta-prune-attribute.csv attribute
```


Confidence Threshold
--------------------

We tune the confidence threshold on the validation set. Repeat the inference steps with test-time augmentation on the our-validation.txt split to get `out-validation-tta-prune-attribute.csv`. Then:

```
python -m xview3.eval.metric --label_file /xview3/all/chips/chip_annotations.csv --scene_path ../data/splits/our-validation.txt --costly_dist --drop_low_detect --inference_file out-validation-tta-prune-attribute.csv --threshold -1
python -m xview3.eval.prune --in_path out-tta-prune-attribute.csv --out_path submit.csv --conf 0.3 # Change to the best confidence threshold.
```


Inquiries
---------

For inquiries, please open a Github issue.
