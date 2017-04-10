# Team Valuca

Vasundhara Gupta
Raluca Niti

Submission for ECSE 415 Dogs and Cats Classification Competition (https://inclass.kaggle.com/c/ecse415-classification/)

## Instructions
Follow instructions to install PyTorch on http://pytorch.org/ (e.g. using pip)

After downloading data from Kaggle (https://inclass.kaggle.com/c/ecse415-classification/), run

`python3 training_segregator.py`

first segregate the data into a directory structure appropriate for the PyTorch ImageFolder Dataset API:

/train
    /cats
    /dogs
/test
    /cats
    /dogs

Then run

`sudo python3 model_trainer.py`

The output will be in `output.csv`