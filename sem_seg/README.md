## Semantic segmentation of indoor scenes

### Dataset

1. Donwload prepared HDF5 data for training:
```
sh +x download_data.sh
```
2. Download 3D indoor parsing dataset (<a href="http://buildingparser.stanford.edu/dataset.html">S3DIS Dataset</a>) for testing and visualization. "Stanford3dDataset_v1.2_Aligned_Version.zip" of the dataset is used. Unzip the downloaded file into "dgcnn/data/", and then run
```
python collect_indoor3d_data.py
```
to generate "dgcnn/data/stanford_indoor3d"

### Train

We use 6-fold training, such that 6 models are trained leaving 1 of 6 areas as the testing area for each model. We keep using 2 GPUs for distributed training. To train 6 models sequentially, run
```
sh +x train_job.sh
```

### Evaluation

1. To generate predicted results for all 6 areas, run 
```
sh +x test_job.sh
```
The model parameters are saved every 10 epochs, the saved model used to generate predited results can be changed by setting "--model_path" in "test_job.sh". For example, if you want to use the model saved after 70 epochs, you can set "--model_path" to "log*n*/epoch_70.ckpt" for *n* = 1, 2, ..., 6. To visualize the results, you can add "--visu" flag in the end of each line in "test_job.sh".

2. To obtain overall quantitative evaluation results, run
```
python eval_iou_accuracy.py
```
