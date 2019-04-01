# Web URL Anomaly Detector
This project details the working of a Neural Network based web anomaly detector for application on URLs. As an overview the system works by passing each URL through 4 probabilistic models, the ouputs from which are fed into a neural network to predict the probability of the URL being an anomaly or non-anomaly. This project was largely inspired by *Krugel* and *Vigina's* work on anomaly detection [1].

---
## Dependencies
- Python 2.7
#### Python Built-in modules
- time
- datetime
- os
- argparse
- random
- math
- pickle
#### Python External modules
Can be downloaded on most systems using `sudo pip install {PACKAGE}`
- tensorflow
- numpy
- keras
- keras_metrics

---
## Dataset
There are several URL log files available openly on the Internet. This log files are usually generated by servers (like Apache). Ideally we download such log files and place them in `Data` folder. Such datasets contain HTTP requests in the form of query along with the access codes. Below is an example of HTTP query in the log file.
```
5.10.83.30 - - [03/Feb/2014:06:28:28 -0600] "GET /jszzang/?document_srl=209013&act=dispMemberLoginForm HTTP/1.1" 400 1020
```
The following commands can be used to download an example dataset.
```
cd Data/
wget http://www.almhuette-raith.at/apache-log/access.log
```

---
## How to run
#### Training
This command uses `access.log` file to train the 4 probabilistic models and neural network model, the trained models of which are stored in the `Logs` folder.
```
python anomaly_detector.py --mode Train --data ./Data/access.log
```
#### Detection
Detection is carried on the file `access.log` to detect the anomalous web URLs using the following command, which are saved in the file `AnomalousURLS.txt` in the `Logs` folder. The `access.log` file is checked for an update and whenever new URLs are logged the detection is carried out again.
```
python anomaly_detector.py --mode Detect --data ./Data/access.log
```

---
## Working
The project contains 4 modules in Modules folder `DataProcessing.py`, `Models.py`, `NeuralNetwork.py`, `Utils.py` which contains different classes and methods used by main file `anomaly_detector.py`.

The main file `anomaly_detector.py` is called when the training and detection module is turned on. The objects of all the four modules from the Modules folder are created in this file and are thus corresponding methods are being used as required.

Also, there are two folders `Logs` and `Data` which contains dataset (log files used for both training and detection) and logs or parameters (generated while running training and detection module) respectively.

#### MODULES

1. DATA PROCESSING : It contains class `DataPreprocessing` which defines methods used for the processing of the data. The individual URL is read from the dataset and a query string identified by leading character ***'?'*** is separated. This query string splits into the attributes and their respective values. Thus, the attributes and their respective values are stored in the form of lists, dictionaries, etc. and further used in training and detection modules. Also, it contains class `DetectionSaver` which defines methods used for saving anomalous URLs and file credentials at the end of the detection.

2. MODELS : It contains 4 classes of attribute models `AttributeLengthModel`, `AttributeCharachterDistributionModel`, `ArrtributeStructuralInference` and `AttributePresenceModel`

  - **Attribute Length Model** : This model focuses on the length of the values of the attributes of n number of URLs in the log file.

    - Training method : In this method the mean and the variance values are calculated from the length of the values of the attributes and stored to be used in detection model.

    - Detection method : In this method the concept of mean and variance to implement ***Chebyshev inequality equation*** is used. With the help of mean, variance values calculated during learning phase and length of values of respective attributes obtained from an individual URL in a log file, the threshold is calculated. Thus URL satisfying the Chebyshev inequality equation is marked as normal while other as anomalous. Also, during training mode the anomaly score depending on the URL marked is returned to form a input dataset to feed it into neural network model for its training.

  - **Attribute Character Distribution Model** : This model deals with the distribution of the characters in the values of the respective attributes for *n* number of URLs in the log file.

    - Training method  : In English grammar the distribution of the characters is nonuniform and with different frequencies. In training method based on character's relative frequency Idealized Character Distribution (ICD) is calculated for individual value of respective attribute. This is also called as expected ICD (E) and is used during detection model.

    - Detection method : This method is based on character's relative frequency, Idealized Character Distribution (ICD) is calculated for individual value of respective attribute of an individual URL called as observed ICD (O). Then the length of the value of the attribute of that URL is multiplied to the expected ICD (E) obtained in the training model. At the end Pearson x-square test is applied with the help of observed ICD calculated during training and expected ICD obtained previously. The value of x-square is then compared to the ***Pearson test*** table. Thus, if the value is greater than `9.2362` then that particular URL is anomalous otherwise vice versa. Also, during training mode the anomaly score depending on the URL marked is returned to form a input dataset to feed it into neural network model for its training.

  - **Attribute Structural Inference** : It depends on the structure of the values of the respective attributes for *n* number of URLs in the log file.

    - Training method : This model uses a ***Markov model*** and ***Bayesian theorem*** for training. After formation of Markov model the transition and emission probabilities are found out for each value of the respective attribute and Bayesian theorem is applied. Based on this theorem among all the Markov model the one yielding maximum likelihood is chosen for the detection model.

    - Detection method : Based on the Markov model obtained from learning model and one of the Bayesian theorem formula, the probability of the value of the respective attribute of an individual URL is obtained. If it is near to `1.0` that means the URL is normal otherwise it is anomalous. Also, during training mode the anomaly score depending on the URL marked is returned to form a input dataset to feed it into neural network model for its training.

  - **Attribute Presence Model** : There might be regularity in number, name, etc. of the attributes.

    - Training method : This model focuses on the presence of the attribute in the URLs. All the attributes are stacked and set is formed which is used during detection phase.

    - Detection method : Individual URL is passed through the detection model and attributes are checked by comparing to attributes set obtained during training phase. If attribute is present that means URL is normal otherwise it is anomalous. Also, during training mode the anomaly score depending on the URL marked is returned to form a input dataset to feed it into neural network model for its training.

3. NEURAL NETWORK : Neural network model is formed by input, output and hidden layers. In supervised machine learning both labeled input and output dataset is used for training and testing. The weights are multiplied to the inputs during training and activation function is used.

  - Training method : During training of this model the number of hidden layers and the number of neurons in those hidden layers, input and output layers are set. The input layer has 4 inputs and the output layer is set to two classes of one hot encoding. Also the rate of alpha and the activation functions are set. Thus, once model is trained then, it is tested on testing dataset (30%). Thus, training and testing goes on till the higher testing accuracy is obtained.

  - Detection method : In the detection method of the neural network each URL is tested whether it is anomalous or not.

4. UTILS : Class for timing methods used to calculate various time features are defined in this module. Also, the methods for checking existence of file and its path and computing overall accuracy, precision, false positive rate are defined in this module.


#### LOGS

Logs and parameters generated while training module are stored here. Also, the file `AnomalousURLs.txt` containing anomalous URLs detected at the end of the detection is stored here.


#### DATA

All datasets (log files) are stored in this directory. Several datasets can exists in this directory simultaneously. Selection to use one of them is done in code.

---
## References
```
[1] Christopher Kruegel and Giovanni Vigna. 2003. Anomaly detection of web-based attacks. In Proceedings of the 10th ACM conference on Computer and communications security (CCS '03). ACM, New York, NY, USA, 251-261. DOI: https://doi.org/10.1145/948109.948144
```