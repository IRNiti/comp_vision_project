# Team Valuca

Vasundhara Gupta
Raluca Niti

Submission for ECSE 415 Dogs and Cats Classification Competition (https://inclass.kaggle.com/c/ecse415-classification/)

## Instructions
1. Follow instructions to install PyTorch on http://pytorch.org/ (e.g. using pip)

2. 
  After downloading data from Kaggle (https://inclass.kaggle.com/c/ecse415-classification/), run

    `python3 training_segregator.py`

first segregate the data into a directory structure appropriate for the PyTorch ImageFolder Dataset API:
  
/train

&nbsp;&nbsp;/cats

&nbsp;&nbsp;/dogs

/test

&nbsp;&nbsp;/cats

&nbsp;&nbsp;/dogs

  Note: The original training data should be in a folder named X_Train.  If using different test data, put the test data in the X_Test folder


Then run

`sudo python3 model_trainer.py`

Note training resnet32 over 5 epochs took 3.5 hours on our CPU.

The output will be in `output.csv`
