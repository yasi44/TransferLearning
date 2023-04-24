# Transfer Learning

## TransferLearning_image.py
Use the pre-trained model 'mobilenet' and retraing it on a set on different flowers.
before retraining process, this model is not able to accuratly detect which type/category each flower is, but after retraining, the model can exactly determine the name of flower.


## TransferLearning_NLP.ipynb
A pretrained model named gnews-swivel-20dim(previousy trained on GNEWS data) is loaded. 
Another layer is added at the end of network to have binary classification on sentiment(positive, negative).
And on the gnew review data of movies, it is retrained.  
here Early Stopping approach is used to prevent overfitting.
