**Introduction**

* This application takes an image of a single digit (see the img directory for examples) 
and determines what number it is. This work is done in two steps with each having their own file.

**Application Details**

1. Create Model (Mnist.py): A Classification Model is made with a Convoluted Neural Network 
(see img/CNN.png for details). The MNIST (Modified National Institute of Standards and Technology) dataset is used to 
train the model, or when the machine "learns".  

2. Image through the Inference Server (DigitClassificationServer.py): A server takes the model when initialized to 
receive images of digits, run them through the Model, and returns the prediction to the client.

**Command Prompt Execution:**

* Model Creation usage: python Mnist.py <Path to Hyperparameters File (Optional)>

* Inference Server usage: python DigitClassificationServer.py

**Implementation Restrictions:**

* The highest accuracy I have seen validating the Model was 96%, not 100%!

* If used, the Hyperparameters file for the Model must be a JSON file.

* For the Hyperparameters allowed, please see the example "hParams.json" file.

* Please do not modify the rows and columns HP if the model will be used for the Inference Server.

* The Model creator will save (or overwrite) the model to the current relative path. It will be named "Mnist.pb".

* The Inference Server will attempt to retrieve "Mnist.pb" from the current relative path. Be sure not to move it.

* All development and testing of this was done in a Windows 10 conda environment. 
I currently have a machine that does not support virtualization, so I am unable to verify that this works for Linux.  

* The Inference Server only uses port 5000 in the localhost, please make sure that is available. 