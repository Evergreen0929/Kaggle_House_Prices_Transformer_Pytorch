# Kaggle_House_Prices_Transformer_Pytorch
A light-weight Transformer model for Kaggle House Prices Regression Competition

# Kaggle House Prices -Advance Regression Techniques

A simple Pytorch deep learning model for predicting the house price. Lightweight Transformer model is tested for accuracy.
![网络结构](https://user-images.githubusercontent.com/90333984/144082762-367e81f0-9e76-4a08-9e97-cf1942b2666a.png)


<!-- # Table of Contents
1. [Project Objective](#objective)
2. [Python Packages](#packages) -->

## Project Objective <a name="p objective"></a>
This is my first project in PyTorch. The aim of the project is to perform a simple multivariate regression using Transformer model. 

## Python Packages
Get packages by using conda or pip.

1. PyTorch=1.8.0
2. numpy=1.19.2
3. matplotlib=3.3.4
4. pandas=1.1.5

## Kaggle
Once finished, you can upload your prediction.csv to the kaggle website where you can compare your score with other users.

Model	5-Fold Validation	Test loss (rmse)
(on official test dataset)
	Train loss (rmse)	Test loss (rmse)	
MLP (1 Block)	0.127530	0.140763	0.15460
MLP (2 Blocks)	0.108675	0.163794	0.15125
Ours	0.017307	0.129986	0.12760


## Acknowledgement
You need to have more than 4GB GPU memory to train the model with default settings, or you need to change batchsize or the network sturctures.
