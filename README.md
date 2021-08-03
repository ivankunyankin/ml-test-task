## Test task for Myna Labs

I decided to use fully convolutional model and PyTorch's implementation of cross-entropy loss


## Running

1. Download and extract the data into the repository's root folder

2. Create an environment  and install the dependencies
``` 
python3 -m venv env 
source env/bin/activate 
pip3 install -r requirements.txt 
```

3. Run the following to start preparing the data:
```
python3 prepare_data.py
```
The code will generate spectrograms and prepare labels for training

4. Run the following to start training:
```
python3 train.py
``` 
Add ```--from_checkpoint``` flag to continue training from a checkpoint if needed.

4. Run the following to start testing:

```
python3 test.py
```

Specify ```--path``` to the .csv containing paths to the testing data

The code will test the trained model on the testing data and save output .csv file into the same directory

5. Tensorboard

```
tensorboard --logdir logs
```
