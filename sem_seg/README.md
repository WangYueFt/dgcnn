## Semantic segmentation of indoor scenes

### Dataset

Donwload prepared HDF5 data for training:
```
sh +x download_data.sh
```
Download 3D indoor parsing dataset (<a href="http://buildingparser.stanford.edu/dataset.html">S3DIS Dataset</a>) for testing and visualization. "Stanford3dDataset_v1.2_Aligned_Version.zip" of the dataset is used. Unzip the downloaded file into "dgcnn/data/", and then run
```
python collect_indoor3d_data.py
```
to generate "dgcnn/data/stanford_indoor3d"
