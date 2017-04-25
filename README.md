# employee-attrition-ml

This project involves building a neural network model to predict employee attrition (whether an employee leaves the company) based on a variety of factors such as age, years worked, and monthly income. 

The publicly available IBM employee attrition dataset is used. A 1-hidden-layer neural network will be built with a softmax one-hot output. 

Exploratory data analysis is performed in IBM Attrition EDA.ipynb and the actual model is built up in network.py.

Running with the default hyperparameters (25 hidden units, 100 mini-batches/epoch, 400 training samples, 0.003 training step) yields a peak accuracy of 86% before dropping off due to overfitting.

Changing the hyperparameters to 50 mini-batches and 0.008 training step yields a peak accuracy of 99%.
