## Run part segmentation experiments

### Requirements

2 GPUs (each with 12 GB memory) for distributed training.

### Data preparation 

Load the data for part segmentation.

```
sh +x download_data.sh
```

### Training 

Train the model using 2 GPUs. 

```
python train_multi_gpu.py
```

Model parameters are saved every 10 epochs in "train_results/trained_models/".

### Testing

To test the model parameter set saved after epoch n, 

```
python test.py --model_path train_results/trained_models/epoch_n.ckpt
```

n = 0, 10, 20, ..., 190, 200.

For example, if we want to test the model saved after 160 epochs, 

```
python test.py --model_path train_results/trained_models/epoch_160.ckpt
```
