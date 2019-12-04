# Risk-Analytics
Home Credit Loan Default Prediction
Home Credit Default Risk
This Project is an extensive study of Feature Tools which is heavily drawn from the featuretools documentation and the featuretools Automated Manual Comparison GitHub repository by Will Koehrsen. 

https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

The Following Report contains detailed description of the entire project and all the steps taken in the codes.

Following content can be found in this report:
1. Project Description
2. Data Description
3. Limitations
4. References
5. Methodologies (with description, code (with comments) and Output 
6. Results
7. Implications 
The objective of this competition is to use historical loan application data to predict whether an applicant will be able to repay a loan. This is a standard supervised classification task:
•	Supervised: The labels are included in the training data and the goal is to train a model to learn to predict the labels from the features
•	Classification: The label is a binary variable, 0 (will repay loan on time), 1 (will have difficulty repaying loan)
Data
The data is provided by Home Credit, a service dedicated to provided lines of credit (loans) to the unbanked population. Predicting whether or not a client will repay a loan or have difficulty is a critical business need, and Home Credit is hosting this competition on Kaggle to see what sort of models the machine learning community can develop to help them in this task.
There are 7 different sources of data:
•	application_train/application_test: the main training and testing data with information about each loan application at Home Credit. Every loan has its own row and is identified by the feature SK_ID_CURR. The training application data comes with the TARGET indicating 0: the loan was repaid or 1: the loan was not repaid.
•	bureau: data concerning client's previous credits from other financial institutions. Each previous credit has its own row in bureau, but one loan in the application data can have multiple previous credits.
•	bureau_balance: monthly data about the previous credits in bureau. Each row is one month of a previous credit, and a single previous credit can have multiple rows, one for each month of the credit length.
•	previous_application: previous applications for loans at Home Credit of clients who have loans in the application data. Each current loan in the application data can have multiple previous loans. Each previous application has one row and is identified by the feature SK_ID_PREV.
•	POS_CASH_BALANCE: monthly data about previous point of sale or cash loans clients have had with Home Credit. Each row is one month of a previous point of sale or cash loan, and a single previous loan can have many rows.
•	credit_card_balance: monthly data about previous credit cards clients have had with Home Credit. Each row is one month of a credit card balance, and a single credit card can have many rows.
•	installments_payment: payment history for previous loans at Home Credit. There is one row for every made payment and one row for every missed payment.




This diagram shows how all of the data is related:

 
 
NOTE: 
Here we cannot obtain a confusion matrix or the Type I and Type II error because the data on which we test is kept hidden from us by Kaggle.
We can evaluate our model on cross validation scores and final submission scores given by Kaggle
Results:
 


This project is divided into 8 parts which are explained below

1. PART 1 – Introduction to the project with EDA and executing Baseline models
2. PART 2 – Manual Feature Engineering
3. PART 3 – Manual Feature Engineering using functions
4. PART 4 – Manual Feature Engineering using functions continued 
5. PART 5 – Automated Feature Engineering
6. PART 6 – Automated Feature Engineering by Tuning Feature Tools
7. PART 7 – Feature Selection
8. PART 8 – Executing PCA and run a basic LGBM model
We followed the general outline of a machine learning project:
1.	Understand the problem and the data
2.	Data cleaning and formatting (this was mostly done for us)
3.	Exploratory Data Analysis
4.	Baseline model
5.	Improved model
6.	Model interpretation
Imports
We are use a typical data science stack: numpy, pandas, sklearn, matplotlib.

 

Read in Data
First, we define the PATH variable where all the available data files are located. There are a total of 9 files: 1 main file for training (with target) 1 main file for testing (without the target), 1 example submission file, and 6 other files containing additional information about each loan.

 
The training data has 307511 observations where each one is a separate loan and 122 features (variables) including the TARGET 

 

The test set is considerably smaller and lacks a TARGET column.

Exploratory Data Analysis
Examine the Distribution of the Target Column
The target is what we are asked to predict: either a 0 for the loan was repaid on time, or a 1 indicating the client had payment difficulties. We first examine the number of loans falling into each category.

 

Here, we see this is an imbalanced class problem. There are far more loans that were repaid on time than loans that were not repaid. Once we get into more sophisticated machine learning models, we weight the classes by their representation in the data to reflect this imbalance.

Examine Missing Values
 we look at the number and percentage of missing values in each column.

 

 

When we build machine learning models, we have to fill in these missing values (known as imputation). In later work, we use models like LightGBM that  handle missing values with no need for imputation. Another option would be to drop columns with a high percentage of missing values, although it is impossible to know ahead of time if these columns be helpful to  model. Therefore, we keep all of the columns for now.

Column Types
We look at the number of columns of each data type. int64 and float64 are numeric variables (which  be either discrete or continuous). Object columns contain strings and are categorical features. 

 

We now look at the number of unique entries in each of the object (categorical) columns.

 

Most of the categorical variables have a relatively small number of unique entries. We find a way to deal with these categorical variables

Label Encoding and One-Hot Encoding
We implement the policy described above: for any categorical variable (dtype == object) with 2 unique categories, we use label encoding, and for any categorical variable with more than 2 unique categories, we use one-hot encoding.
For label encoding, we use the Scikit-Learn LabelEncoder and for one-hot encoding, the pandas get_dummies(df) function.

 

Aligning Training and Testing Data
There  be same features (columns) in both the training and testing data. One-hot encoding has created more columns in the training data because there were some categorical variables with categories not represented in the testing data. To remove the columns in the training data that are not in the testing data, we align the dataframes. First we extract the target column from the training data (because this is not in the testing data but we keep this information). When we do the align, we must make sure to set axis = 1 to align the dataframes based on the columns and not on the rows

 

The training and testing datasets now have the same features which is required for machine learning. The number of features has grown signifitly due to one-hot encoding. Later we    try dimensionality reduction (removing features that are not relevant) to reduce the size of the datasets.

Back to Exploratory Data Analysis
Anomalies
One problem we always have to be on the lookout for when doing EDA is anomalies within the data. These be due to mis-typed numbers, errors in measuring equipment, or they could be valid but extreme measurements. One way to support anomalies quantitatively is by looking at the statistics of a column using the describe method. The numbers in the DAYS_BIRTH column are negative because they are recorded relative to the current loan application. To see these stats in years, we  multiply by -1 and divide by the number of days in a year:

 

Those ages look reasonable. There are no outliers for the age on either the high or low end. Now we look at days of employment

 

That doesn't look right. The maximum value (besides being positive) is about 1000 years

 
We subset the anomalous clients and see if they tend to have higher or low rates of default than the rest of the clients.

 

It is interesting that the anomalies have a lower rate of default.
Handling the anomalies depends on the exact situation, with no set rules. One of the safest approaches is just to set the anomalies to a missing value and then have them filled in (using Imputation) before machine learning. In this case, since all the anomalies have the exact same value, we fill them in with the same value in case all of these loans share something in common. The anomalous values seem to have some importance, so we tell the machine learning model if we did in fact fill in these values. As a solution, we fill in the anomalous values with not a number (np.nan) and then create a new Boolean column indicating whether or not the value was anomalous.

 

The distribution looks to be much more in line with what we would expect, and we also have created a new column to tell the model that these values were originally anomalous (because we have to fill in the nans with some value,  the median of the column). The other columns with DAYS in the dataframe look to be about what we expect with no obvious outliers.
As an extremely important note, anything we do to the training data we also have to do to the testing data. We make sure to create the new column and fill in the existing column with np.nan in the testing data.

 

We look at some of more significant correlations: the DAYS_BIRTH is the most positive correlation. (except for TARGET because the correlation of a variable with itself is always 1) Looking at the documentation, DAYS_BIRTH is the age in days of the client at the time of the loan in negative days. The correlation is positive, but the value of this feature is negative, meaning that as the client gets older, they are less likely to default on their loan (i.e. the target == 0). That's a little confusing, so we take the absolute value of the feature and then the correlation be negative.

Effect of Age on Repayment

 

As the client gets older, there is a negative linear relationship with the target meaning that as clients get older, they tend to repay their loans on time more often.
We start looking at this variable. First, we make a histogram of the age. We put the x axis in years to make the plot a little more understandable.

 

By itself, the distribution of age does not tell us much other than that there are no outliers as all the ages are reasonable. To visualize the effect of the age on the target, we make a kernel density estimation plot (KDE) colored by the value of the target. A kernel density estimate plot shows the distribution of a single variable and  be thought of as a smoothed histogram (it is created by computing a kernel, usually a Gaussian, at each data point and then averaging all the individual kernels to develop a single smooth curve). We use the seaborn kdeplot for this graph.

 

 

The target == 1 curve skews towards the younger end of the range. Although this is not a significant correlation (-0.07 correlation coefficient), this variable is likely going to be useful in a machine learning model because it does affect the target. We look at this relationship in another way: average failure to repay loans by age bracket.
To make this graph, first we cut the age category into bins of 5 years each. Then, for each bin, we calculate the average value of the target, which tells us the ratio of loans that were not repaid in each age category.
In [24]:
 

 

 

 

There is a clear trend: younger applicants are more likely to not repay the loan. The rate of failure to repay is above 10% for the youngest three age groups and below 5% for the oldest age group.
This is information that could be directly used by the bank: because younger clients are less likely to repay the loan, maybe they  be provided with more guidance or financial planning tips. This does not mean the bank  discriminate against younger clients, but it would be smart to take precautionary measures to help younger clients pay on time.

Exterior Sces
The 3 variables with the strongest negative correlations with the target are EXT_SCE_1, EXT_SCE_2, and EXT_SCE_3. According to the documentation, these features represent a "normalized score from external data sce". We are not sure what this exactly means, but it be a cumulative sort of credit rating made using numerous sces of data.
We take a look at these variables.
First, we show the correlations of the EXT_SCE features with the target and with each other.

 

 

All three EXT_SCE features have negative correlations with the target, indicating that as the value of the EXT_SCE increases, the client is more likely to repay the loan. We also see that DAYS_BIRTH is positively correlated with EXT_SCE_1 indicating that maybe one of the factors in this score is the client age.
We look at the distribution of each of these features colored by the value of the target. This lets us visualize the effect of this variable on the target.

 

 
 

EXT_SCE_3 displays the greatest difference between the values of the target. We clearly see that this feature has some relationship to the likelihood of an applicant to repay a loan. The relationship is not very strong (in fact they are all considered very weak, but these variables  still be useful for a machine learning model to predict whether or not an applicant repay a loan on time.

Jake VanderPlas writes about polynomial features in his excellent book Python for Data Science for those who want more information.
In the following code, we create polynomial features using the EXT_SCE variables and the DAYS_BIRTH variable. Scikit-Learn has a useful class called PolynomialFeatures that creates the polynomials and the interaction terms up to a specified degree. We use a degree of 3 to see the results (when we create polynomial features, we avoid using too high of a degree, both because the number of features scales exponentially with the degree, and because we can run into problems with overfitting).

 

This creates a considerable number of new features. To get the names we must use the polynomial features get_feature_names method.

 

There are 35 features with individual features raised to powers up to degree 3 and interaction terms. Now, we see whether any of these new features are correlated with the target.

 

Several of the new variables have a greater (absolute magnitude) correlation with the target than the original features. When we build machine learning models, we try with and without these features to determine if they actually help the model learn.
We add these features to a copy of the training and testing data and then evaluate models with and without the features.

 

Domain Knowledge Features
we make a couple features that attempt to capture what we think be important for telling whether a client  default on a loan. Here we are going to use five features that were inspired by this script by Aguiar:
•	CREDIT_INCOME_PERCENT: the percentage of the credit amount relative to a client's income
•	ANNUITY_INCOME_PERCENT: the percentage of the loan annuity relative to a client's income
•	CREDIT_TERM: the length of the payment in months (since the annuity is the monthly amount due
•	DAYS_EMPLOYED_PERCENT: the percentage of the days employed relative to the client's age
Again, thanks to Aguiar and his great script for exploring these features.

 

Visualize New Variables
We  explore these domain knowledge variables visually in a graph. For all of these, we make the same KDE plot colored by the value of the TARGET.

  
 
 

It's hard to say ahead of time if these new features be useful.

Baseline
If we guess 1 value of Target for all observations on the test set. This get us a Receiver Operating Characteristic Area Under the Curve (AUC ROC) of 0.5 in the competition.
Since we already know what score we are going to get, we don't make a naive baseline guess. We use a slightly more sophisticated model for  actual baseline: Logistic Regression.
Logistic Regression Implementation
To get a baseline, we use all of the features after encoding the categorical variables. We preprocess the data by filling in the missing values (imputation) and normalizing the range of the features (feature scaling).

 

We  use LogisticRegression from Scikit-Learn for  first model. The only change we make from the default model settings is to lower the regularization parameter, C, which controls the amount of overfitting (a lower value  decrease overfitting). This got us slightly better results than the default LogisticRegression.
Here we use the familiar Scikit-Learn modeling syntax: we first create the model, then we train the model using .fit and then we make predictions on the testing data using .predict_proba

 

Now that the model has been trained, we use it to make predictions. We predict the probabilities of not paying a loan, so we use the model predict.proba method. This returns an m x 2 array where m is the number of observations. The first column is the probability of the target being 0 and the second column is the probability of the target being 1 (so for a single row, the two columns must sum to 1). We want the probability the loan is not repaid, so we select the second column.

 

The predictions must be in the format shown in the sample_submission.csv file, where there are only two columns: SK_ID_CURR and TARGET. We create a dataframe in this format from the test set and the predictions called submit.

 

The predictions represent a probability between 0 and 1 that the loan not be repaid. If we were using these predictions to classify applicants, we set a probability threshold for determining that a loan is risky.

 

The submission has now been saved to the virtual environment in which  part is running. To access the submission, at the end of the part, we hit the blue Commit & Run button at the upper right of the kernel. This runs the entire part and then lets us download any files that are created during the run.
Once we run the part, the files created are available in the Versions tab under the Output sub-tab. From here, the submission files be submitted to the competition or downloaded. Since there are several models in this project, there are multiple output files.
The logistic regression baseline scored around 0.671 when submitted.
Improved Model: Random Forest
To try and beat the poor performance of  baseline, we update the algorithm. We try using a Random Forest on the same training data to see how that affects performance. The Random Forest is a much more powerful model especially when we use hundreds of trees. We use 100 trees in the random forest.

 

This model scored around 0.678 when submitted.
Making Predictions using Engineered Features
The only way to see if the Polynomial Features and Domain knowledge improved the model is to train a test a model on these features, We then compare the submission performance to that for the model without these features to gauge the effect of  feature engineering.

 

This model scored 0.678 when submitted to the competition, exactly the same as that without the engineered features. Given these results, it does not appear that feature construction helped in this case.
Testing Domain Features
Now we test the domain features made manually.

 

This scored 0.679 when submitted which shows that the engineered features do not help in this model.

Model Interpretation: Feature Importances
As a simple method to see which variables are the most relevant, we look at the feature importances of the random forest. Given the correlations we saw in the exploratory data analysis, we expect that the most important features are the EXT_SCE and the DAYS_BIRTH. We use these feature importances as a method of dimensionality reduction in future work.


 

 

As expected, the most important features are those dealing with EXT_SCE and DAYS_BIRTH. We see that there are only a handful of features with a significant importance to the model, which suggests we are able to drop many of the features without a decrease in performance (and we even see an increase in performance.) Feature importances are not the most sophisticated method to interpret a model or perform dimensionality reduction, but they make us start to understand what factors model takes into account when it makes predictions.

 



Light Gradient Boosting Machine
We step off the deep end and use a real machine learning model: the gradient boosting machine using the LightGBM library The Gradient Boosting Machine is currently the leading model for learning on structured datasets (especially on Kaggle).
         
  
 
This submission scored about 0.735 on the leaderboard.
   
 
Again, we see that some of  features made it into the most important. Going forward, we think about what other domain knowledge features be useful for this problem (or we can consult someone who knows more about the financial industry)
 
This model scored 0.754 when submitted to the public leaderboard indicating that the domain features do improve the performance. Feature engineering is going to be a critical part of this competition (as it is for all machine learning problems)
###################################### PART 2
Here, we look at using information from the bureau and bureau_balance data. The definitions of these data files are:
•	bureau: information about client's previous loans with other financial institutions reported to Home Credit. Each previous loan has its own row.
•	bureau_balance: monthly information about the previous loans. Each month has its own row.
 
To illustrate the general process of manual feature engineering, we first simply get the count of a client's previous loans at other financial institutions. This requires a number of Pandas operations we make heavy use of throughout the part:
•	groupby: group a dataframe by a column. In this case we group by the unique client, the SK_ID_CURR column
•	agg: perform a calculation on the grouped data such as taking the mean of columns. We  either call the function directly (grouped_df.mean()) or use the agg function together with a list of transforms (grouped_df.agg([mean, max, min, sum]))
•	merge: match the aggregated statistics to the appropriate client. We merge the original training data with the calculated stats on the SK_ID_CURR column which  insert NaN in any cell for which the client does not have the corresponding statistic
We also use the (rename) function quite a bit specifying the columns to be renamed as a dictionary. This is useful in order to keep track of the new variables we create.
This is a lot, which is why we eventually write a function to do this process for us. We take a look at implementing this by hand first.
 
 
 

Assessing Usefulness of New Variable with r value
To determine if the new variable is useful, we calculate the Pearson Correlation Coefficient (r-value) between this variable and the target. This measures the strength of a linear relationship between two variables and ranges from -1 (perfectly negatively linear) to +1 (perfectly positively linear). The r-value is not best measure of the "usefulness" of a new variable, but it gives a first approximation of whether a variable can be helpful to a machine learning model. The larger the r-value of a variable with respect to the target, the more a change in this variable is likely to affect the value of the target. Therefore, we look for the variables with the greatest absolute value r-value relative to the target.
We also visually inspect a relationship with the target using the Kernel Density Estimate (KDE) plot.
Kernel Density Estimate Plots
The kernel density estimate plot shows the distribution of a single variable (think of it as a smoothed histogram). To see the different in distributions dependent on the value of a categorical variable, we color the distributions differently according to the category. For example, we show the kernel density estimate of the previous_loan_count colored by whether the TARGET = 1 or 0. The resulting KDE show any significant differences in the distribution of the variable between people who did not repay their loan (TARGET == 1) and the people who did (TARGET == 0). This serves as an indicator of whether a variable will 'relevant' to a machine learning model.
We put this plotting functionality in a function to re-use for any variable.
 
We test this function using the EXT_SCE_3 variable which we found to be one of the most important variables according to a Random Forest and Gradient Boosting Machine.
 
Now for the new variable we just made, the number of previous loans at other institutions.
 
From this it's difficult to tell if this variable be important. The correlation coefficient is extremely weak and there is almost no noticeable difference in the distributions.
We move on to make a few more variables from the bureau dataframe. We take the mean, min, and max of every numeric column in the bureau dataframe.

Aggregating Numeric Columns
To account for the numeric information in the bureau dataframe, we compute statistics for all the numeric columns. To do so, we groupby the client id, agg the grouped dataframe, and merge the result back into the training data. The agg function  only calculate the values for the numeric columns where the operation is considered valid. We  stick to using 'mean', 'max', 'min', 'sum' but any function  be passed in here. We even write own function and use it in an agg call.
 
We create new names for each of these columns. The following code makes new names by appending the stat to the name. Here we must deal with the fact that the dataframe has a multi-level index.
 
Now we simply merge with the training data as we did before.
 

Correlations of Aggregated Values with Target
We calculate the correlation of all new values with the target. Again, we use these as an approximation of the variables which be important for modeling.
 
In the code below, we sort the correlations by the magnitude (absolute value) using the sorted Python function. We also make use of an anonymous lambda function.
 
None of the new variables have a significant correlation with the TARGET. We look at the KDE plot of the highest correlated variable, bureau_DAYS_CREDIT_mean, with the target in in terms of absolute magnitude correlation.
 
The definition of this column is: "How many days before current application did client apply for Credit Bureau credit". Our interpretation is this is the number of days that the previous loan was applied for before the application for a loan at Home Credit. Therefore, a larger negative number indicates the loan was further before the current loan application. We see an extremely weak positive relationship between the average of this variable and the target meaning that clients who applied for loans further in the past potentially are more likely to repay loans at Home Credit. With a correlation this weak though, it is just as likely to be noise as a signal.
The Multiple Comparisons Problem
When there are lots of variables, we can expect some of them to be correlated just by pure chance, a problem known as multiple comparisons. We make hundreds of features, and some turn out to be corelated with the target simply because of random noise in the data. Then, when model trains, it overfit to these variables because it thinks they have a relationship with the target in the training set, but this does not necessarily generalize to the test set. There are many considerations that we must take into account when making features
Function for Numeric Aggregations
We encapsulate all the previous work into a function. This allows us to compute aggregate stats for numeric columns across any dataframe. We re-use this function when we apply the same operations for other dataframes.
   
 
To make sure the function worked as intended, we compare with the aggregated dataframe we constructed by hand.
 

If we go through and inspect the values, we do find that they are equivalent. We are able to reuse this function for calculating numeric stats for other dataframes. Using functions allows for consistent results and decreases the amount of work we must do in the future

Correlation Function
Before we move on, we also make the code to calculate correlations with the target into a function.
 
First, we one-hot encode a dataframe with only the categorical columns (dtype == 'object').
 
 

The sum columns represent the count of that category for the associated client and the mean represents the normalized count. One-hot encoding makes the process of calculating these figures very easy
We use a similar function as before to rename the columns. Again, we must deal with the multi-level index for the columns. We iterate through the first level (level 0) which is the name of the categorical variable appended with the value of the category (from one-hot encoding). Then we iterate stats we calculated for each client. We rename the column with the level 0 name appended with the stat. As an example, the column with CREDIT_ACTIVE_Active as level 0 and sum as level 1 become CREDIT_ACTIVE_Active_count.
 
 
 

The sum column records the counts and the mean column records the normalized count.
We merge this dataframe into the training data.
 
 
Function to Handle Categorical Variables
To make the code more efficient, we write a function to handle the categorical variables for us. This take the same form as the agg_numeric function in that it accepts a dataframe and a grouping variable. Then it calculates the counts and normalized counts of each category for all categorical variables in the dataframe.
 
 

Applying Operations to another dataframe
We now turn to the bureau balance dataframe. This dataframe has monthly information about each client's previous loan(s) with other financial institutions. Instead of grouping this dataframe by the SK_ID_CURR which is the client id, we first group the dataframe by the SK_ID_BUREAU which is the id of the previous loan. This gives us one row of the dataframe for each loan. Then, we group by the SK_ID_CURR and calculate the aggregations across the loans of each client. The final result be a dataframe with one row for each client, with stats calculated for their loans.
 
First, we calculate the value counts of each status for each loan. We already made a function that does this for us
 
Now we handle the one numeric column. The MONTHS_BALANCE column has the "months of balance relative to application date." This might not necessarily be that important as a numeric variable, and in future work we might consider this as a time variable. For now, we calculate the same aggregation statistics as previously.
 
The above dataframes have the calculations done on each loan. Now we aggregate these for each client. We do this by merging the dataframes together first and then since all the variables are numeric, we just aggregate the statistics again, this time grouping by the SK_ID_CURR.
 
 

Review, for the bureau_balance dataframe we:
1.	Calculated numeric stats grouping by each loan
2.	Made value counts of each categorical variable grouping by loan
3.	Merged the stats and the value counts on the loans
4.	Calculated numeric stats for the resulting dataframe grouping by the client id
The final resulting dataframe has one row for each client, with statistics calculated for all of their loans with monthly balance information.
Some of these variables are a little confusing, so We try to explain a few:
•	client_bureau_balance_MONTHS_BALANCE_mean_mean: For each loan calculate the mean value of MONTHS_BALANCE. Then for each client, calculate the mean of this value for all of their loans.
•	client_bureau_balance_STATUS_X_count_norm_sum: For each loan, calculate the number of occurences of STATUS == X divided by the number of total STATUS values for the loan. Then, for each client, add up the values for each loan.
We hold off on calculating the correlations until we have all the variables together in one dataframe.

Putting the Functions Together
We now have all the pieces in place to take the information from the previous loans at other institutions and the monthly payments information about these loans and put them into the main training dataframe. We do a reset of all the variables and then use the functions we built to do this from the ground up. This demonstrates the benefit of using functions for repeatable workflows
 
 
Counts of Bureau Dataframe
 
Aggregated Stats of Bureau Dataframe
 
Value counts of Bureau Balance dataframe by loan
 
Aggregated stats of Bureau Balance dataframe by loan
 
Aggregated Stats of Bureau Balance by Client
 
Insert Computed Features into Training Data
 
 
 

Feature Engineering Outcomes
Now we take a look at the variables we have created. We look at the percentage of missing values, the correlations of variables with the target and the correlation of variables with the other variables. The correlations between variables show if we have collinear variables, that is, variables that are highly correlated with one another. Often, we remove one in a pair of collinear variables because having both variables would be redundant. We also use the percentage of missing values to remove features with a substantial majority of values that are not present.
Feature selection is the process of removing variables to help model to learn and generalize better to the testing set. The objective is to remove useless/redundant variables while preserving those that are useful. There are a number of tools we use for this process, but in this part we  stick to removing columns with a high percentage of missing values and variables that have a high correlation with one another.

Missing Values
An important consideration is the missing values in the dataframe. Columns with too many missing values might have to be dropped.
 
 
We see there are several columns with a high percentage of missing values. There is no well-established threshold for removing missing values, and the best case of action depends on the problem. Here, to reduce the number of features, we remove any columns in either the training or the testing data that have greater than 90% missing values.
 
Before we remove the missing values, we find the missing value percentages in the testing data. We then remove any columns with greater than 90% missing values in either the training or testing data. We now read in the testing data, perform the same operations, and look at the missing values in the testing data. We already have calculated all the counts and aggregation statistics, so we only merge the testing data with the appropriate data.

Calculate Information for Testing Data
 
 
We align the testing and training dataframes, which means matching up the columns so they have the exact same columns. This shouldn't be an issue here, but when we one-hot encode variables, we align the dataframes to make sure they have the same columns.
 
 
The dataframes now have the same columns (with the exception of the TARGET column in the training data). This means we can use them in a machine learning model which needs to see the same columns in both the training and testing dataframes.
We now look at the percentage of missing values in the testing data so we figure out the columns that  be dropped.
 
 

We ended up removing no columns in this round because there are no columns with more than 90% missing values. We might have to apply another feature selection method to reduce the dimensionality.
At this point we save both the training and testing data. I encage anyone to try different percentages for dropping the missing columns and compare the outcomes.

 

Correlations
We look at the correlations of the variables with the target. We see in any of the variables we created have a greater correlation than those already present in the training data (from application).

 
 

The highest correlated variable with the target (other than the TARGET which of case has a correlation of 1), is a variable we created. However, just because the variable is correlated does not mean that it is useful, and we have to remember that if we generate hundreds of new variables, some are going to be correlated with the target simply because of random noise.
Viewing the correlations skeptically, it does appear that several of the newly created variables be useful. To assess the "usefulness" of variables, we look at the feature importances returned by the model. we make a kde plot of two of the newly created variables.

 

This distribution is all over the place. This variable represents the number of previous loans with a CREDIT_ACTIVE value of Active divided by the total number of previous loans for a client. The correlation here is so weak that we do not think we can draw any conclusions

Collinear Variables
We calculate not only the correlations of the variables with the target, but also the correlation of each variable with every other variable. This allows us to see if there are highly collinear variables that  perhaps be removed from the data.
We look for any variables that have a greater than 0.8 correlation with other variables.

 

For each of these pairs of highly correlated variables, we only remove one of the variables. The following code creates a set of variables to remove by only adding one of each pair.

 

We remove these columns from both the training and the testing datasets. We have to compare performance after removing these variables with performance keeping these variables (the raw csv files we saved earlier).

   

Modeling
•	control: only the data in the application files.
•	test one: the data in the application files with all of the data recorded from the bureau and bureau_balance files
•	test two: the data in the application files with all of the data recorded from the bureau and bureau_balance files with highly correlated variables removed.
 

  

 

Control
The first step in any experiment is establishing a control. For this we use the function defined above (that implements a Gradient Boosting Machine model) and the single main data source (application).



 

Fortunately, once we have taken the time to write a function, using it is simple (In this project, we use functions to make things simpler and reproducible). The function above returns a submission dataframe we upload to the competition, a fi dataframe of feature importances, and a metrics dataframe with validation and test performance.
 
 
The control slightly overfits because the training score is higher than the validation score. We address this in later when we look at regularization (we already perform some regularization in this model by using reg_lambda and reg_alpha as well as early stopping).
We visualize the feature importance with another function, plot_feature_importances. The feature importances be useful when it's time for feature selection.
 
 
The control scores 0.745 when submitted to the competition.
Test One
We conduct the first test. We just  pass in the data to the function, which does most of the work for us.
 
 
Based on these numbers, the engineered features perform better than the control case. However, we have to submit the predictions to the leaderboard before we say if this better validation performance transfers to the testing data.
 
Examining the feature importances, it looks as if a few of the feature we constructed are among the most important. We find the percentage of the top 100 most important features that we made in this part. However, rather than just compare to the original features, we compare to the one-hot encoded original features. These are already recorded for us in fi (from the original data).
 
Over half of the top 100 features were made by us. That  gave us confidence that all the hard work we did was worthwhile.
 
Test one scored 0.759 when submitted to the competition.
Test Two
That was easy, so We do another run Same as before but with the highly collinear variables removed.
  
  
Test Two scored 0.753 when submitted to the competition.

Results
 
###################################### PART 3
Here use the aggregation and value counting functions developed in that part in order to incorporate information from the previous_application, POS_CASH_balance, installments_payments, and credit_card_balance data files. We already used the information from the bureau and bureau_balance in the previous parts and were able to improve competition score compared to using only the application data. After running a model with the features included here, performance does increase, but we run into issues with an explosion in the number of features We worked on a part of feature selection, but for this part we continue building up a rich set of data for model.
The definitions of the additional data files are:
•	previous_application (called previous): previous applications for loans at Home Credit of clients who have loans in the application data. Each current loan in the application data have multiple previous loans. Each previous application has one row and is identified by the feature SK_ID_PREV.
•	POS_CASH_BALANCE (called cash): monthly data about previous point of sale or cash loans clients have had with Home Credit. Each row is one month of a previous point of sale or cash loan, and a single previous loan have many rows.
•	credit_card_balance (called credit): monthly data about previous credit cards clients have had with Home Credit. Each row is one month of a credit card balance, and a single credit card have many rows.
•	installments_payment (called installments): payment history for previous loans at Home Credit. There is one row for every made payment and one row for every missed payment.
Functions

•	agg_numeric: calculate aggregation statistics (mean, count, max, min) for numeric variables.
•	agg_categorical: compute counts and normalized counts of each category in a categorical variable.
Together, these two functions extract information about both the numeric and categorical data in a dataframe.  general approach be to apply both of these functions to the dataframes, grouping by the client id, SK_ID_CURR. For the POS_CASH_balance, credit_card_balance, and installment_payments, we first group by the SK_ID_PREV, the unique id for the previous loan. Then we group the resulting dataframe by the SK_ID_CURR to calculate the aggregation statistics for each client across all of their previous loans.

 

Function to Aggregate Numeric Data
This groups data by the group_var and calculates mean, max, min, and sum. It can only be applied to numeric data by default in pandas.
 


Function to Calculate Categorical Counts
This function calculates the occurrences (counts) of each category in a categorical variable for each client. It also calculates the normed count, which is the count for a category divided by the total counts for all categories in a categorical variable.

 


Function for KDE Plots of Variable
We also made a function that plots the distribution of variable colored by the value of TARGET (either 1 for did not repay the loan or 0 for did repay the loan). We use this function to visually examine any new variables we create. This also calculates the correlation coefficient of the variable with the target which be used as an approximation of whether or not the created variable be useful.

 

Function to Convert Data Types
This help reduce memory usage by using more efficient types for the variables. For example, category is often a better type than object (unless the number of unique categories is close to the number of rows in the dataframe).

 
We deal with one dataframe at a time. First up is the previous_applications. This has one row for every previous loan a client had at Home Credit. A client has multiple previous loans which is why we aggregate statistics for each client.

previous_application

 

 
 

We join the calculated dataframe to the main training dataframe using a merge. Then we delete the calculated dataframes to avoid using too much of the kernel memory.
 

We don't want to overwhelm the model with too many irrelevant features or features with too many missing values. In the previous part, we removed any features with more than 75% missing values. To be consistent, we apply that same logic here.

Function to Calculate Missing Values

 

 

 

Applying to More Data

Function to Aggregate Stats at the Client Level

 

Monthly Cash Data

 
  

 

Monthly Credit Data

 

 

 

Installment Payments

 

 
 

Save All Newly Calculated Features
Code and data is uploaded online you find the entire datasets here. 
 

Modeling
 




###################################### PART 4
We were not able to execute part 3 because of memory error so we continue here
 
 
 

The control scored 0.77052 when submitted to the competition.







###################################### PART 5

In this part, we apply automated feature engineering using the featuretools library. Featuretools is an open-source Python package for automatically creating new features from multiple tables of structured, related data. It is ideal tool for problems such as the Home Credit Default Risk competition where there are several related tables that need to be combined into a single dataframe for training (and one for testing).

Read in Data and Create Small Datasets
We read in the full dataset, sort by the SK_ID_CURR and keep only the first 1000 rows to make the calculations feasible. Later we can convert to a script and run with the entire datasets.

 

We join the train and test set together but add a separate column identifying the set. This is important because we want to apply the same exact procedures to each dataset. It's safest to just join them together and treat them as a single dataframe.
(We are not sure if this is allowing data leakage into the train set and if these feature creation operations should be applied separately. Any thoughts would be much appreciated!)
 

Featuretools Basics
Featuretools is an open-source Python library for automatically creating features out of a set of related tables using a technique called deep feature synthesis. Automated feature engineering, like many topics in machine learning, is a complex subject built upon a foundation of simpler ideas.
There are a few concepts that we cover:
•	Entities and EntitySets
•	Relationships between tables
•	Feature primitives: aggregations and transformations
•	Deep feature synthesis
Entities and Entitysets
An entity is simply a table or in Pandas, a dataframe. The observations are in the rows and the features in the columns. An entity in featuretools must have a unique index where none of the elements are duplicated. Currently, only app, bureau, and previous have unique indices (SK_ID_CURR, SK_ID_BUREAU, and SK_ID_PREV respectively). For the other dataframes, we pass in make_index = True and then specify the name of the index. Entities can also have time indices where each entry is identified by a unique time.
An EntitySet is a collection of tables and the relationships between them. This can be thought of a data structure with its own methods and attributes. Using an EntitySet allows us to group together multiple tables and manipulate them much quicker than individual tables.
First, we make an empty entityset named clients to keep track of all the data.
 
Now we define each entity, or table of data. We pass in an index if the data has one or make_index = True if not. Featuretools will automatically infer the types of variables, but we can also change them if needed.
 

Relationships
Relationships are a fundamental concept not only in featuretools, but in any relational database. The way to think of a one-to-many relationship is with the analogy of parent-to-child. A parent is a single individual but can have multiple children. The children can then have multiple children of their own. In a parent table, each individual has a single row. Each individual in the parent table can have multiple rows in the child table.
For example, the app dataframe has one row for each client (SK_ID_CURR) while the bureau dataframe has multiple previous loans (SK_ID_PREV) for each parent (SK_ID_CURR). Therefore, the bureau dataframe is the child of the app dataframe. The bureau dataframe in turn is the parent of bureau_balance because each loan has one row in bureau but multiple monthly records in bureau_balance.
 
 

The SK_ID_CURR "100002" has one row in the parent table and multiple rows in the child.
Two tables are linked via a shared variable. The app and bureau dataframe are linked by the SK_ID_CURR variable while the bureau and bureau_balance dataframes are linked with the SK_ID_BUREAU. Defining the relationships is relatively straightforward, and the diagram provided by the competition is helpful for seeing the relationships. For each relationship, we need to specify the parent variable and the child variable. Altogether, there are a total of 6 relationships between the tables. Below we specify all six relationships and then add them to the EntitySet.
 
 
 


Note: Need to be careful to not create a diamond graph where there are multiple paths from a parent to a child. If we directly link `app` and `cash` via `SK_ID_CURR`; `previous` and `cash` via `SK_ID_PREV`; and `app` and `previous` via `SK_ID_CURR`, then we have created two paths from `app` to `cash`. This results in ambiguity, so the approach we have to take instead is to link `app` to `cash` through `previous`. We establish a relationship between `previous` (the parent) and `cash` (the child) using `SK_ID_PREV`. Then we establish a relationship between `app` (the parent) and `previous` (now the child) using `SK_ID_CURR`. Then featuretools will be able to create features on `app` derived from both `previous` and `cash` by stacking multiple primitives.

All entities in the entity can be related to each other. In theory this allows to calculate features for any of the entities, but in practice, we only calculate features for the `app` dataframe since that will be used for training/testing.

Feature Primitives
A feature primitive is an operation applied to a table or a set of tables to create a feature. These represent simple calculations, many of which we already use in manual feature engineering (Part 3, 4, 5), that can be stacked on top of each other to create complex features. Feature primitives fall into two categories:
•	Aggregation: function that groups together child datapoints for each parent and then calculates a statistic such as mean, min, max, or standard deviation. An example is calculating the maximum previous loan amount for each client. An aggregation works across multiple tables using relationships between tables.
•	Transformation: an operation applied to one or more columns in a single table. An example would be taking the absolute value of a column or finding the difference between two columns in one table.
A list of the available features primitives in featuretools is viewed below.

  

 
 
Deep Feature Synthesis
Deep Feature Synthesis (DFS) is the process featuretools uses to make new features. DFS stacks feature primitives to form features with a "depth" equal to the number of primitives. For example, if we take the maximum value of a client's previous loans (say MAX(previous.loan_amount)), that is a "deep feature" with a depth of 1. To create a feature with a depth of two, we could stack primitives by taking the maximum value of a client's average monthly payments per previous loan (such as MAX(previous(MEAN(installments.payment)))). Here is the original paper on automated feature engineering using deep feature synthesis.
To perform DFS in featuretools, we use the dfs function passing it an entityset, the target_entity (where we want to make the features), the agg_primitives to use, the trans_primitives to use and the max_depth of the features. Here we will use the default aggregation and transformation primitives, a max depth of 2, and calculate primitives for the app entity. Because this process is computationally expensive, we run the function using features_only = True to return only a list of the features and not calculate the features themselves. This is useful to look at the resulting features before starting an extended computation.
  
 

 

To apply this on entire dataset, Unfortunately, will not run in a personal computer due to the computational expense of the operation. Using a computer with 64GB of ram, this function call takes around 24 hours. The entire dataset available here in the file called feature_matrix.csv.

DFS with Selected Aggregation Primitives
With featuretools, we were able to go from 121 original features to almost 1700 in a few lines of code. When we did feature engineering by hand, it took about 12 hours to create a comparable size dataset. However, while we get a lot of features in featuretools, this function call is not very well-informed. We simply used the default aggregations without thinking about which ones are "important" for the problem. We ended up with a lot of features, but they are probably not all relevant to the problem. Too many irrelevant features can decrease performance by drowning out the important features (related to the curse of dimensionality)
The next call we make will specify a smaller set of features. We still don’t use much domain knowledge, but this feature set is more manageable. The next step is improving the features we actually build and performing feature selection.
 
That gave us 884 features (and took about 12 hours to run on the complete dataset).
 
The following correlations were calculated using the entire training section of the feature_matrix_spec.
 

We explore the Correlations with the Target
 
 

We look for pairs of correlated features and potentially remove any above a threshold of 90%.
 

 
 
 
 

 
 
 
 
 

###################################### PART 6
 
Read in Data and Create Smaller Datasets
We limit the data to 1000 rows because automated feature engineering is computationally intensive work. Later we can refactor this code into functions and put it in a script to run on a more powerful machine.
 
Properly Representing Variable Types
There are a number of columns in the app dataframe that are represented as integers but are really discrete variables that can only take on a limited number of features. Some of these are Boolean flags (only 1 or 0) and two columns are ordinal (ordered discrete). To tell featuretools to treat these as Boolean variables, we need to pass in the correct datatype using a dictionary mapping {variable_name: variable_type}.
 
There are also two ordinal variables in the app data: the rating of the region with and without the city.
 
The previous data also has two Boolean variables.
 
Although we do not know the actual application date, if we assume a starting application date that is the same for all clients, then we can convert the MONTHS_BALANCE into a datetime. This can then be treated as a relative time that we can use to find trends or identify the most recent value of a variable.
Replace Outliers
There are a number of day offsets that are recorded as 365243. Reading through discussions, others replaced this number with np.nan. If we don't do this, Pandas will not be able to convert into a timedelta and throws an error that the number is too large. The following code has been adapted from a script on GitHub.
 
First, we establish an arbitrary date and then convert the time offset in months into a Pandas timedeltaobject.
 
 
These four columns represent different offsets:
•	DAYS_CREDIT: Number of days before current application at Home Credit client applied for loan at other financial institution. We call this the application date, bureau_credit_application_date and make it the time_index of the entity.
•	DAYS_CREDIT_ENDDATE: Number of days of credit remaining at time of client's application at Home Credit. We call this the ending date, bureau_credit_end_date
•	DAYS_ENDDATE_FACT: For closed credits, the number of days before current application at Home Credit that credit at other financial institution ended. We call this the closing date, bureau_credit_close_date.
•	DAYS_CREDIT_UPDATE: Number of days before current application at Home Credit that the most recent information about the previous credit arrived. We call this the update date, bureau_credit_update_date.
 
Plot for a sanity check
To make sure the conversion went as planned, we make a plot showing the distribution of loan lengths.
 
There are a number of loans that are unreasonably long. Reading through the discussions, other people had noticed this as well. At this point, we just leave in the outliers. We also drop the time offset columns.
Bureau Balance
The bureau balance dataframe has a MONTHS_BALANCE column that we can use as a months offset. The resulting column of dates can be used as a time_index.
 
Previous Applications
The previous dataframe holds previous applications at Home Credit. There are a number of time offset columns in this dataset:
•	DAYS_DECISION: number of days before current application at Home Credit that decision was made about previous application. This represent the time_index of the data.
•	DAYS_FIRST_DRAWING: number of days before current application at Home Credit that first disbursement was made
•	DAYS_FIRST_DUE: number of days before current application at Home Credit that first due was suppoed to be
•	DAYS_LAST_DUE_1ST_VERSION: number of days before current application at Home Credit that first was??
•	DAYS_LAST_DUE: number of days before current application at Home Credit of last due date of previous application
•	DAYS_TERMINATION: number of days before current application at Home Credit of expected termination
We convert all these into timedeltas in a loop and then make time columns.
 
Previous Credit and Cash
The credit_card_balance and POS_CASH_balance each have a MONTHS_BALANCE column with the month offset. This is the number of months before the current application at Home Credit of the previous application record. These represent the time_index of the data.
 
Installments Payments
The installments_payments data contains information on each payment made on the previous loans at Home Credit. It has two date offset columns:
•	DAYS_INSTALMENT: number of days before current application at Home Credit that previous installment was supposed to be paid
•	DAYS_ENTRY_PAYMENT: number of days before current application at Home Credit that previous installment was actually paid
The process is: convert to timedeltas and then make time columns. The DAYS_INSTALMENT will serve as the time_index.
 
Applying Featuretools
We now start making features using the time columns. We create an entityset named clients much as before, but now we have time variables that we can use.
 
Relationships
The relationships between tables has not changed since the previous implementation.
 
Time Features
We look at Time features we can make from the new time variables. Because these times are relative and not absolute, we are only interested in values that show change over time, such as trend or cumulative sum. We would not want to calculate values like the year or month since we choose an arbitrary starting date.
We pass in a chunk_size to the dfs call which specifies the number of rows (if an integer) or the fraction or rows to use in each chunk (if a float). This can help to optimize the dfs procedure, and the chunk_size can have a significant effect on the run time. Here we use a chunk size equal to the number of rows in the data so all the results will be calculated in one pass. We also avoid making any features with the testing data, so we pass in ignore_entities = [app_test].
 
We visualize one of these new variables. We can look at the trend in credit size over time. A positive value indicates that the loan size for the client is increasing over time.
   
Interesting Values
Another method we use in featuretools is "interesting values." Specifying interesting values calculates new features conditioned on values of existing features. For example, we can create new features that are conditioned on the value of NAME_CONTRACT_STATUS in the previous dataframe. Each stat will be calculated for the specified interesting values which can be useful when we know that there are certain indicators that are of greater importance in the data.
 
We assign interesting values to the variable and then specify the where_primitives in the dfs call.
 
One of the features is MEAN(previous.CNT_PAYMENT WHERE NAME_CONTRACT_STATUS = Approved). This shows the average "term of previous credit" on previous loans conditioned on the previous loan being approved. We compare the distribution of this feature to the MEAN(previous.CNT_PAYMENT WHERE NAME_CONTRACT_STATUS = Canceled) to see how these loans differ.

 
 
Based on the most important features returned by a model, we can create new interesting features. This is the area where domain knowledge can be applied to feature creation.
Seed Features
An additional extension to the default aggregations and transformations is to use seed features. These are user defined features that we provide to deep feature synthesis that can then be built on top of where possible.
As an example, we can create a seed feature that determines whether or not a payment was late. This time when we make the dfs function call, we need to pass in the seed_features argument.
 
 
Another seed feature we use is whether or not a previous loan at another institution was past due.
 
Create Custom Feature Primitives
we can also write our own Feature Primitives. This is an extremely powerful method that lets us expand the capabilities of featuretools.
NormalizedModeCount and LongestSeq
As an example, we make three features, building on code from the featuretools GitHub. These will be aggregation primitives, where the function takes in an array of values and returns a single value. The first, NormalizedModeCount, builds upon the Mode function by returning the fraction of total observations in a categorical feature that the model makes up. In other words, for a client with 5 total bureau_balance observations where 4 of the STATUS were X, the value of the NormalizedModeCount would be 0.8. The idea is to record not only the most common value, but also the relative frequency of the most common value compared to all observations.
The second custom feature will record the longest consecutive run of a discrete variable. LongestSeq takes in an array of discrete values and returns the element that appears the most consecutive times. Because entities in the entityset are sorted by the time_index, this will return the value that occurs the most number of times in a row with respect to time.
   
 
   
MostRecent
The final custom feature is MOSTRECENT. This simply returns the most recent value of a discrete variable with respect to time columns in a dataframe. When we create an entity, featuretools will sort the entity by the time_index. Therefore, the built-in aggregation primitive LAST calculates the most recent value based on the time index. However, in cases where there are multiple different time columns, it might be useful to know the most recent value with respect to all of the times. To build the custom feature primitive, We adapted the existing TREND primitive (code here).
 
To test whether this function works as intended, we can compare the most recent variable of CREDIT_TYPE ordered by two different dates.

 
For client 100002
Most recent type of credit was Credit card if we order by the application date, but Consumer credit if we order by the end date of the loan
Putting it all Together
Finally, we run deep feature synthesis with the time variables, with the correct specified categorical variables, with the interesting features, with the seed features, and with the custom features. To actually run this on the entire dataset, we can take the code here, put it in a script, and then use more computational resources.

 
 


###################################### PART 7
 
* train_bureau is the training features built manually using the bureau and bureau_balance data
* train_previous is the training features built manually using the previous, cash, credit, and installmentsdata
We first see how many features we built over the manual engineering process. Here we use a couple of set operations to find the columns that are only in the bureau, only in the previous, and in both dataframes, indicating that there are original features from the application dataframe. Here we work with a small subset of the data in order to not overwhelm the machine. This code has also been run on the full dataset (we will take a look at some of the results).
 
That gives us the number of features in each dataframe. Now we combine the data without creating any duplicate rows.
 
When we did this to the full dataset, we get 1465 features.
Correcting Mistakes
When we were doing manual feature engineering, we accidentally created some columns derived from the client id, SK_ID_CURR. As this is a unique identifier for each client, it should not have any predictive power, and we would not want to build a model trained on this "feature". So, we remove any columns built on the SK_ID_CURR.
 
After applying this to the full dataset, we end up with 1416 features. More features might seem like a good thing, and they can be if they help our model learn. However, irrelevant features, highly correlated features, and missing values can prevent the model from learning and decrease generalization performance on the testing data. Therefore, we perform feature selection to keep only the most useful variables. We start feature selection by focusing on collinear variables.
Remove Collinear Variables
Collinear variables are those which are highly correlated with one another. These can decrease the model's availablility to learn, decrease model interpretability, and decrease generalization performance on the test set, these are three things we want to increase, so removing collinear variables is a useful step. We establish an admittedly arbitrary threshold for removing collinear variables, and then remove one out of any pair of variables that is above that threshold.
The code below identifies the highly correlated variables based on the absolute magnitude of the Pearson correlation coefficient being greater than 0.9. Again, this is not entirely accurate since we are dealing with such a limited section of the data.
This code is adapted from work by Chris Albon.
Identify Correlated Variables
 
 

Drop Correlated Variables

 
Applying this on the entire dataset results in 538 collinear features removed.
This has reduced the number of features singificantly, but it is likely still too many. At this point, we read in the full dataset after removing correlated variables for further feature selection.
The full datasets (after removing correlated variables) are available in m_train_combined.csv and m_test_combined.csv.
Read in Full Dataset
Now we move on to the full set of features. These were built by applying the above steps to the entire train_bureau and train_previous files

 
Remove Missing Values
A relatively simple choice of feature selection is removing missing values. We have to decide what percentage of missing values is the minimum threshold for removing a column. Like many choices in machine learning, there is no right answer, and not even a general rule of thumb for making this choice. In this implementation, if any columns have greater than 75% missing values, they will be removed.
Most models (including those in Sk-Learn) cannot handle missing values, so we have to fill these in before machine learning. The Gradient Boosting Machine ((in LightGBM)) can handle missing values. Imputing missing values always makes it little uncomfortable because we add information that actually isn't in the dataset.

 

We drop the columns, one-hot encode the dataframes, and then align the columns of the dataframes.

 
One method for doing this automatically is the Recursive Feature Elimination method in Scikit-Learn.
Instead of doing this automatically, we perform our own feature removal by first removing all zero importance features from the model. If this leaves too many features, then we consider removing the features with the lowest importance. We use a Gradient Boosted Model from the LightGBM library to assess feature importances. If Scikit-Learn library is used, the LightGBM library has an API that makes deploying the model very similar to using a Scikit-Learn model.

 
 
One of feature made it into the top 5 most important. That's a good sign our hard work making the features. It also looks like many of the features we made have literally 0 importance. For the gradient boosting machine, features with 0 importance are not used at all to make any splits. Therefore, we remove these features from the model with no effect on performance (except for faster training).

   
   
We remove the features that have zero importance.
 
   
 
There are now no 0 importance features left. If we desire to remove more features, we have to start with features that have a non-zero importance. One way we could do this is by retaining enough features to account for a threshold percentage of importance, such as 95%. At this point, we keep enough features to account for 95% of the importance. Again, this is an arbitrary decision

   
We keep only the features needed for 95% importance. This step seems to have the greatest chance of harming the model's learning ability, so rather than changing the original dataset, we make smaller copies. Then, we test both versions of the data to see if the extra feature removal step is worthwhile.


 

Test New Featuresets
The last step of feature removal we did, may have the potential to hurt the model the most. Therefore we test the effect of this removal. To do that, we use a standard model and change the features.
We use a fairly standard LightGBM model, similar to the one we used for feature selection. The main difference is this model uses five-fold cross validation for training and we use it to make predictions. There's a lot of code here, but that's because we included documentation and a few extras (such as feature importances) that aren't strictly necessary. We use the same model with two different datasets to see which one performs the best.

 
Test "Full" Dataset
This is the expanded dataset. The process used to make this dataset is:
•	Removed collinear features as measured by the correlation coefficient greater than 0.9
•	Removed any columns with greater than 80% missing values in the train or test set
•	Removed all features with non-zero feature importances
   
 

The full features after feature selection scored 0.783 when submitted to the public leaderboard.
Test "Small" Dataset
The small dataset does one additional step over the full dataset:
•	Keep only features needed to reach 95% cumulative importance in the gradient boosting machine
 
 
The smaller featureset scored 0.782 when submitted to the public leaderboard.

Conclusions
In this part we employed a number of feature selection methods. These methods are necessary to reduce the number of features to increase model interpretability, decrease model runtime, and increase generalization performance on the test set. The methods of feature selection we used are:

1. Remove highly collinear variables as measured by a correlation coefficient greater than 0.9
2. Remove any columns with more than 75% missing values.
3. Remove any features with a zero importance as determined by a gradient boosting machine.
4. Keep only enough features to account for 95% of the importance in the gradient boosting machine.

Using the first three methods, we reduced the number of features from __1465__ to __536__ with a 5-fold cv AUC ROC score of 0.7838 and a public leaderboard score of 0.783.

After applying the fourth method, we end up with 342 features with a 5-fold cv AUC SCORE of 0.7482 and a public leaderboard score of 0.782.
###################################### PART 8
 
 
PCA
We can select the number of new components, or the fraction of variance we want explained in the data. If we pass in no argument, the number of principal components will be the same as the number of original features. We can then use the variance_explained_ratio_ to determine the number of components needed for different threshold of variance retained.

 
We need a few principal components to account for the majority of variance in the data. We use the first two principal components to visualize the entire dataset.

 
 
  
 
   
We color the datapoints by the value of the target to see if using two principal components clearly separates the classes.
 
 
 
Even though we accounted for most of the variance, that does not mean the pca decomposition makes the problem of identifying loans repaid vs not repaid any easier. PCA does not consider the value of the label when projecting the features to a lower dimension.
The model scores 0.77396 when submitted to the competition.
PCA does not consider the value of the label when projecting the features to a lower dimension
