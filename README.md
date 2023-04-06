# DLCV05



#### Install required libraries

pip install -r requirements.txt



## Demo

The training are visualised by TensorBoard, the log data is saved in ./.logs/


#### Training

Check all of the input arguments of train.py

```
python train.py --help
```

Run train.py, for example

```
python train.py --model net7 --dropout --batchnorm --lr_scheduler --save net7dbs
```

You also can run fulltrain.py with full training dataset, without cross-validation, for example

```
python fulltrain.py --model net7 --dropout --batchnorm --lr_scheduler --save net7dbs
```


#### Test

Trained models are automatically saved in ./checkpoints/, 

Run test.py by selecting trained model and config file for it, for example

```
python test.py --model checkpoints/net7dbs/099.pkl --configs checkpoints/net7dbs/configs.json
```
