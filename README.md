# Neural_Network_Charity_Analysis
Building a Deep Learning Model to Categorize and Predict Successful Non-Profits. 

## Overview 
In order to help predict the likelyhood of a successful non-profit, AlphabetSoup had requested help in developing a Neural Network model. The model will classify future non-profits applying for funding from AlphabetSoup to either successful or not-successful categories, based on prior data. The [Dataset](https://github.com/Fabalin/Neural_Network_Charity_Analysis/blob/main/Resources/charity_data.csv) includes data on 34,300 Non-profit organizations that have applied to AlphabetSoup across these characteristics: 

- **EIN**-Unique Identification number
- **NAME**—Name of charity 
- **APPLICATION_TYPE**-Type of Application used by the non-profit
- **AFFILIATION**—Affiliated sector of industry
- **CLASSIFICATION**—Government organization classification
- **USE_CASE**—Intended Use 
- **ORGANIZATION**—Organization type
- **STATUS**—Active status
- **INCOME_AMT**—Income classification
- **SPECIAL_CONSIDERATIONS**—Special consideration for application
- **ASK_AMT**—Funding amount requested
- **IS_SUCCESSFUL**—Success or failure of the non-profit 

The dataset was cleaned and analyzed using scikit-learn and the neural network was built using tensorflow. The model's predictive accuracy was assessed and numerous manual attempts were made to optimize the model and 3 of these attempts were reported. All 3 attempts can be seen in the version history of [AlphabetSoupCharity_Optimization.ipynb](https://github.com/Fabalin/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.ipynb) file. The acurracy of the final model was 62% with a 0.67% loss. The full model is available for download as [AlphabetSoupCharity_Optimization.h5](https://github.com/Fabalin/Neural_Network_Charity_Analysis/blob/main/AlphabetSoupCharity_Optimization.h5) 

## Software
- Python 3.7.13
- Jupyter Notebook v6.4.11

### Dependencies
- pandas 1.4.2
- numpy 1.21.5
- sklearn 1.0.2
- tensorflow 2.9.1

## Results 
### Data Preprocessing 
- The target variable is the **IS_SUCCESSFUL** column in the dataset and the values are a binary of 0-False or 1-True. 

- Within the remaining columns, **EIN** and **NAME** were discarded since they were identification columns and **STATUS** was also discarded since it virtually had no variation within the data. 

- The lefover columns were determined to be the features after being binned and one-hot encoded.  

### Compiling, Training and Evaluating the Model
- The final model had 3 hidden layers with 1 output layer. The number of Neurons and the activation functions are displayed below: 
  
  <img width="524" alt="image" src="https://user-images.githubusercontent.com/99558296/176981840-0c9ac271-64a3-4168-94f3-7f2ac0865492.png">    
  
  <img width="329" alt="image" src="https://user-images.githubusercontent.com/99558296/176981868-e37184f0-c6ed-420c-807e-28cd89ecec5d.png">
  
  These values were determined through trial and error and were designed to expand the analysis of the input features by doubling the neuron amount and funneling towards smaller amounts of neurons. The ReLU activation function is crucial as it enables non-linear input data to be analyzed effectively. 

- The target model performance of >75% was not achieved. The highest accuracy observed was 62%. 

- Although previous versions of the optimization model with smaller neurons in the hidden layers could perform at 53% accuracy, the results were often more chaotic. The neurons of the final model were . Previous iterations of the model had incorporated the tanh activation function in the hidden layers to facilitate higher changes in weight due to having a greater gradient than the sigmoid function. However, substituting ReLu function in place of the tanh, yielded consistently higher accuracy results. The sample parameters along with its associated accuracy score is shown below from the first uploaded version of the optimized file: 
  
  <img width="364" alt="image" src="https://user-images.githubusercontent.com/99558296/176983927-bb84f2aa-cecf-4942-b37e-60af5ea85d4e.png">                      
  
  <img width="400" alt="image" src="https://user-images.githubusercontent.com/99558296/176983939-a354549d-84b2-4401-bd03-0455a61619c1.png">

- The next step towards optimizing the model was limiting the amount of epochs to have an efficient training process that would provide optimal loss without sacrificing computing power and risk overtraining. The loss of the model was visualized across 20, 50 and 200 epochs and convergence of loss and accuracy was observed around 20 - 30 epochs. Thus, future models were limited to 30 epochs. The image files for the loss graphs are located in the [Resources](https://github.com/Fabalin/Neural_Network_Charity_Analysis/blob/main/Resources/loss_20epochs.png) folder. The most common accuracy score of the model evaluation was around 53%. The images below demonstrate change in loss over 200, 50 and 20 epochs respectively: 
  ![image](https://user-images.githubusercontent.com/99558296/176984043-3a563850-0f22-4249-9fe3-be65768cdb1d.png)
  ![image](https://user-images.githubusercontent.com/99558296/176984050-a1b0df98-0899-4db4-83e9-f5cb384eafa0.png)
  ![image](https://user-images.githubusercontent.com/99558296/176984056-c6e4e092-8dd7-44ec-ba99-468d1b36ab42.png)

- Finally, the input data was then re-processed to eliminate noise, previous iterations of this involved: dropping the **SPECIAL_CONSIDERATION** due to its low amount of variation. Binning for the rest of the columns were also refined to eliminate granularity within the input features that did not provide much information. All these attempts would yield minimal results and thus were omitted. The final model would be fed the categorical data while dropping the negligable variation seen in the Other values of the **USE_CASE** and **AFFILIATION** features. Binning these values along with the other values within the same column produced no noticeable differece. This they were omitted prior to splitting the data into training and testing sets, after one-hot encoding:
  <img width="368" alt="image" src="https://user-images.githubusercontent.com/99558296/176983778-243b0a3c-45cf-4f29-a0db-ee658c7020c6.png">
  <img width="302" alt="image" src="https://user-images.githubusercontent.com/99558296/176983793-2bc865de-8295-4a5c-83d0-48325c757bcd.png">
  
 ## Summary 
 The optimized deep learning model had 62% accuracy upon evaluation with the test dataset. The model was optimized manually over multiple attempts with the final model training over 30 epochs with ReLU activated hidden layers and a final Sigmoid activated output layer using the hyper parameters below: 
 
<img width="355" alt="image" src="https://user-images.githubusercontent.com/99558296/176984215-0fc226bb-f0bb-42ec-924f-d76293a1bbea.png">        

Although the model could not achieve the target predictive accuracy of 75% the optimization process could be automated with the [keras-tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) library. Due to the volume of features, it is more efficient to use a neural-network model that incorporates non-linear functions rather than a linear regression to categorize the results. Multiple attempts with a maximum of 5 hidden layers were performed but to no significant improvements in accuracy. Since optimization could occur across multiple hyperparameters, it is more efficient to automate the process especially if the user is unfamiliar with the underlying algorithms that power the neural network by transforming each input. The most impact towards improving the accuracy in this instance came from optimization through data pre-processing to significantly eliminate noise without sacrificing the variablity. An alternative model using variable amounts of tanh and ReLU activation functions across optimized amounts of hidden layers and neurons could yield better results than simply limiting the model to ReLU functions alone. This could help fully characterize the non-linear patterns within the data and increase the model's sensitivity. This analysis could not provide a definitively optimized model as there are many hyper parameter variations left to be tested. 
 


 

  
  


