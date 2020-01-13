# American-Sign-language-prediction-using-CNN
Model was developed to understand signs of American Sign Language into corresponding letters in English with help of Convolution Neural Network based classifier. The dataset had images of signs were captured in 5 different sessions, with similar lighting and background. Suitable CNN was built to do classification and TensorFlow was used for this project and achieved an accuracy of 96%. 


American Sign Language (ASL) is a natural language that serves as the predominant sign language of Deaf communities in the United States and most of Anglophone Canada (More about this language can be found here at this Link (Links to an external site.)).

In this project I got a chance to help the community by developing a system to interpret the ASL signs into corresponding English letters. It is a multi-class classification problem, but I built a Convolution Neural Network based classifier using TensorFlow. I have decided to build the classifier to be able to classify only the first 9 letters: 'A', 'B', 'C', 'D', 'E', 'F','G','H','I'.

Dataset
Dataset A (Easy)" prepared by Nicolas Pugeault and Richard Bowden from their Center for Vision, Speech and Signal Processing lab at University of Surrey, UK. 
Please download only the "Dataset A: 5 users (easy)" from this link [ Link (Links to an external site.) ]. The archive size is ~2.2GB. Please don't get scared, you are only going to use a subset of it.
This dataset comprises 24 static signs (excluding letters j and z because they involve motion and can't be represented in an image). This was captured in 5 different sessions, with similar lighting and background. You are going to use only the dataset relevant for classifying the first 9 letters.
Split the dataset into training and test (80%, 20% respectively), and use the training set for training the CNN classifier, and finally use the test set to test your model.
Build a suitable CNN to be able to do the classification well. Please play with the CNN structure (i.e., number of filters, size of filters, padding, number of convolution layers, fully connected layers, pooling dimensions, number of epochs, etc..).
