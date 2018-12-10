## Part segmentation

### Dataset 

Load the data for part segmentation.

```
sh +x download_data.sh
```

### Train

Train the model on 2 GPUs, each with 12 GB memeory. 

```
python train_multi_gpu.py
```

Model parameters are saved every 5 epochs in "train_results/trained_models/".

### Evaluation

To evaluate the model saved after epoch n, 

```
python test.py --model_path train_results/trained_models/epoch_n.ckpt
```

For example, if we want to test the model saved after 175 epochs (provided), 

```
python test.py --model_path train_results/trained_models/epoch_175.ckpt
```
