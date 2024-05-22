### Project Title
Anomaly Detection using Machine Learning (ML) Models to Accelerate Bug Finding During Validation

**Author**

Sanjay Ravi

#### Executive summary
This research work addresses the problem of anomaly detection during hardware/system validation, which involves how to accelerate bug finding through models that predict if a test sequence is normal or anomalous.  A few machine learning (ML) classfication models were created, tested, and compared against each other in order to determine how accurately these models would predict anomalies.  Two of these models have been showing prediction accuracy of 94%, while the third model has been showing a reduced prediction accuracy of 82%.  One of these models has been able to identify the most important token ID's for the validation teams to investigate for debugging these anomalies and to root cause the source of the problem.

#### Rationale
Without answering the research question below, the hardware/system validation teams will not be able to identify anomalies or bugs quickly, and also will not be able to gain quick insights into what types of sequences or patterns cause these anomalies or bugs to occur within the hardware/system.  This will slow down the debugging required to ship the product out the door to the customer, negatively impacting the hardware business.

#### Research Question
How do we find anomalies during hardware or hardware/software system validation in order to accelerate bug finding for faster time-to-market?

#### Data Sources
Data source is related to LogBERT-HDFS from Github:
https://github.com/HelenGuohx/logbert

#### Methodology
Scripts from the Github link above were used to create the dataset with the test sequences containing lists of token IDs.  Then unsupervised machine learning was used to create the normal and anomaly labels, since in the real world the dataset to create the models from might not have pre-defined class labels.  Then natural language processing (NLP) based feature extraction was done to process the dataset so that it could be consumed by the ML models.  Two standard classification algorithms as well as neural network were used to validate the prediction capability.

Results will be how good the prediction is, using the test portion of the dataset, which will predict what types of sequences in the hardware/system validation will be normal and which will be anomalous.

#### Results
The summary of results are:
* Both Decision Tree and Logistic Regression have equal and high test accuracy (94% accuracy), while neural network has lower accuracy of 82%.
* For Decision Tree and Logistic Regression models, precision is high but recall is low. This means that false predictions of anomalies is low, but false predictions of normal is high.
* Advantage of neural network is for applications where recall needs to be high, since results show high recall for the neural network. This is when it is very important to have false predictions of normal to be low. However, the drawback of neural network is lower accuracy and precision compared to the other 2 models.
* Main advantage of Decision Tree model is the speed, where mean fit time is less.
* For the validation users, recommendation would be to use precision as metric, since false predictions of anomalies needs to be low in order to minimize resources used to debug anomalies which might not be a problem.
* Recommendation would be to use the Logistic Regression model, since this has highest precision score of all 3 models, and also most important features to analyze can be easily extracted.
* The most important features to analyze while debugging the anomalies are these token IDs
	* token ID 16
	* token ID 26
	* token ID 44
	* token ID 38
	* token ID 25

#### Next steps
* Important for the system validation user to check for the words or tokens corresponding to the above token IDs within the original logfile in order to debug the source of the anomalies or bugs.
* Also important for the system validation user to refine the model by retraining with additional data on a periodic basis (once every few weeks, months, etc.). New sequences not in the current model that lead to bugs could come up in the future.
* User to try a prediction by entering a sequence of token IDs on the streamlit website.  The ML model runs on this website, and it will provide a prediction of either normal or anomaly.  One of the Jupyter notebooks below will provide a pointer on how to run this streamlit app.

#### Outline of project

- Link to Jupyter Notebook 1: (Main notebook with Decision Tree, Logistic Regression, and comparison of all 3 models)
[https://github.com/SanjayRRavi/Module24_Capstone_Phase2/blob/main/Capstone_Phase2.ipynb]

- Link to Jupyter Notebook 2: (Secondary notebook with neural network model done in Google Colab)
  

- Link to Jupyter Notebook 3:


#### Contact and Further Information

Sanjay Ravi

email: sanjay.ravi1975@gmail.com
