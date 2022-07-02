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

  These values were determined through trial and error. Although previous versions of the optimization model with smaller neurons in the hidden layers could perform at 53% accuracy, the results were often more chaotic. The neurons of the final model were designed to expand the analysis of the input features by doubling the neuron amount and funneling towards smaller amounts of neurons. Previous iterations of the model had incorporated the tanh activation function in the 2nd hidden layer 


