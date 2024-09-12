# Real-Time Myocardial Infarction(MI) Prediction

The objective of this project is to develop a real-time system that can predict the likelihood of a person experiencing a Myocardial Infarction (one kind of  heart attack) using big data tools and machine learning.

By analyzing the person’s ECG data in real-time, the system aims to provide early warnings to healthcare professionals, allowing them to take prompt action and provide better medical care to potentially prevent or reduce the severity of heart attacks.

The dataset used for this project contains 1D ECG data.

The goal of this project is to utilize Kafka to stream data from a CSV file to Spark, where the machine learning model performed real-time predictions.

![image](https://github.com/user-attachments/assets/002bbb6b-0a77-4ab5-8416-053f624f1037)

## Dataset

* The PTB Diagnostics dataset consists of ECG records from 290 subjects.
* This dataset is open source and freely available on Kaggle. The link to the dataset is
  * https://www.kaggle.com/datasets/shayanfazeli/heartbeat
* The data has been preprocessed to 
  * handle missing values
  * all the samples have been cropped, downsampled and padded with zeroes to make its length equal to a predefined fixed length.
 
* In the dataset
  * Number of Samples: 14552.
  * Number of Categories: 2.
  * Sampling Frequency: 125Hz.
  * Data Source: Physionet's PTB Diagnostic Database.

  * 148 diagnosed as MI , 52 healthy control, and the rest are diagnosed with 7 different disease. 
  * Each record contains ECG signals from 12 leads sampled at the frequency of 1000Hz. 
  * In this study we have only used ECG lead II and worked with MI and healthy control categories in our analyses.

![image](https://github.com/user-attachments/assets/6a421c48-36e0-4e07-9fa8-2125564609e8)

