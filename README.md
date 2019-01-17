# Forecasting-Customer-Loan-Default
Used to Forecasting Customer Loan Default Using Ensemble Learning

 
Group Project: Phase 1
Group name: Blue Team

# Project Title: Forecasting Customer loan Default using Ensemble Learning

# Introduction

Modern retail banking follows a simple business rule: Gather cash from depositors and invest it into riskier assets. These riskier assets include securities, companies and government debt as well as loans. Loans companies were historically using ratio analysis such as income to loan and income to interest to makes eligibility decisions. The recent growth of Machine learning has seen a direct application to the loaning market, and numerous studies have been made such as Kumar et al. (2018) and Arutjothi et al. (2017). We will expand on those studies using the knowledge we acquired in this class.


# Description of the dataset

Lending Club (no date) datasets is a comprehensive credit data for all loans issued within the time 2007 to 2011 period. The datasets include the current loan status (Current, Late, Fully Paid, etc.) and the earliest payment report. The file containing loan data through the "present" contains complete loan data for all loans issued through the previous completed calendar quarter. The dataset has a Data Dictionary that includes descriptions for all the data attributes.


# Why we choose Lending Club dataset

This dataset is very exhaustive and offers us the opportunity to apply the knowledge we gathered from previous classes. We will be able to use data preprocessing methods, dimensionality reduction as well as clustering, classification, and Deep Learning techniques.

We also find the idea of developing an application to forecasting Loan Eligibility thrilling as it is a real-world technology that can benefit both applicants and Lenders.

# Benefits of our application

Applicants will be able quickly to check whether they will pass the eligibility rate in our models without impacting their credit score.
Lenders will benefit from a higher accuracy at forecasting loan default and can adjust their lending rate to reflect the increased risk of the individual, or flatly refuse the application.


# Risk assessment

Some models might contain bias against people based on their sex, gender, race, ethnicity, age, religion and a whole lot of other things that we probably never meant to do. The bias happens because the correlations between all the variables in our data set are never exactly zero. Everything is correlated with everything else to some degree and if we are fighting for every last decimal predictive ability, we are going to run into these things, and accidentally build bias into our models.
Now, what did we do about it?  We make sure that the Lending Club data complies with federal regulations. It uses anonymized data where information without protected categories and identifiers are removed. Making sure the algorithms and our models to use data appropriately. Making sure they're focusing on the variables that we need and that you're not accidentally introducing associated variables that bring in bias. 

# Literature review

Historically, Loan Eligibility was (and still is depending on the lender) determined using a metric system (scorecard) based on ratio analysis such as the Debt to Income Ratio (DTI), employment status and available capital amongst other (Thomas, 2000). As statistics grow, numerous studies starting using statistical techniques such as logit regression (Hyeongjun et al, 2018) to model loan default. Nowadays, more advanced methods called for a machine learning model. Kumar et al. (2018) use a wide variety of machine learning method to predict loan approval, but he does not present objective model evaluation. On the other hand, other studies have focused on only one learning method such as Amira et al (2013) who used Neural Network to accurately forecast loan default or Arutjothi et al (2017) who relied on the K-NN classifier to predict loan default (using the dataset source as us). We will extend to these studies by answering the following questions:

	-Are missing information such as job title correlated with loans default

-Which features commonly used by the industry (DTI, employment, etc.…) are a correct predictor of loan delinquency? Are there other relevant features that should be considered?

-Does clustering our dataset and training our machine learning model on those clusters provides better accuracy than training our machine learning model on the whole dataset? (Das et al., 2017   uses clustering methods before training his machine learning model)

-Which machine learning model provides better accuracy? 

- Is Ensemble learning a better approach?

We will be using the following machine learning Model for clustering:

-	PAM: these clustering techniques use swap neighborhood operation and medoids to determine the optimal cluster point (Reynolds et a., 2014). It deals better with outliers than the k-means methodology thanks to his usage of medoids.

# We will be using the following machine learning Model for classification:

-	K Nearest Neighbors: this simple technique that classifies a point basis the most present class around that point.
-	Decision Trees: this technique uses information entropy methods to classify a point by creating a series of nodes representing a series of if/else statement. Decision Trees are known as a weak classifier and can overfit data easily. For this reason, we will combine it with the adaboost method to create a stronger predictor. 
-	Random Forest: this technique is a combination of decision trees based that are calibrated different random subset of the original dataset. This enhances the stability and accuracy of the model (Maroco et al,2011)
-	Naïve Bayes: Naives Bayes similarly approach the dataset to K Nearest Neighbors by classifying a point basis the class around it. However, it weights these classes by their overall representation on the dataset, making it less sensitive to the overrepresentation of a class on the dataset. However, Naïve bayes assume naively that the features are independent
-	Deep Feed Forward Network: Deep Feed Forward Network has been used successfully to classify loan application with a high degree of accuracy (Amira et al, 2013). They rely on a complex mathematical logic that mimics the brain
-	Linear Regression: this method is powerful when there is a linear relationship between variables.
-	SVM: Support Vector Machine is a method that used linear classification to maps points in space. It creates a hyperplane which maximizes the distance between the points of different classes and classify new from that hyperplane.
-	Kernel SVM: Kernel is similar to the SVM, except that it used the Kernel trick to transform a linear space into a non-linear space
-	Ensemble Learning: Ensemble Learning will combine all the previous model into a weighted average for classification purpose. 
-	

# Project Step/Tasks List

 	 Comment	Deadline	Responsible
  
Step 1 - Data Cleaning
Sub Category: Missing Data	All the missing data are non-numerical: We will first test if missing attributes are correlated via chi-square test with the classification attribute we are trying to estimate (i.e. Loan Status) to determine whether they are MCAR, MAR or MNAR. We will likely replace the missing value with a global constant indicating that they did not provide a description and keep an eye out for MNAR categories	 
Sub Category: Categorical Features	We need to replace Categorical Features with numerical values. One Hot Encoder is the technique we are thinking about using	 
			
Step 2 - Feature Engineering and Dimensionality Reduction via Feature Selection 
Sub Category: Feature Engineering	We will create new features basis our literature review. For example, debt to income ratio is a feature we will create	 
Sub Category: Feature Redundancy	We will first test for features redundancy using Person Correlation Matrix and remove feature with correlation above a certain threshold 	 
Sub Category: Features Elimination	We will select the key features using Recursive Feature Elimination via Random Forest	 
			
Step 3- Training Set and Test Set
Sub Category: Creating the Training Set	We will split up the dataset into a Training Set and a Test Set using	 
			
Step 4- Clustering
Sub Category: Determining the optimal number of Clusters	We will create multiple dataset. The first dataset will contain all the data. The other datasets will be created using clustering methodology. we will determine the optimal number of clusters by combining the elbow method and the silhouette measure. We will see if clustering improving the accuracy of our model.	 	 
Sub Category: Finalising the new dataset	We will create the new datasets based on the output from PAM using the optimal number of clusters above.
			
Step 4- Normalising Dataset
Sub Category: Normalising Dataset	Some machine learning model need to be trained on normalized data. We will use Gaussian normalization	 
			
Step 5- Machine Learning
Sub Category: K Nearest Neighbours	 	 
Sub Category: Decisions Trees with Ada Boost	  	
Sub Category: Random Forest	 	 
Sub Category: Naïve Bayes	 	 
Sub Category: Deep Feed Forward Neural Network	 	 
Sub Category: Linear Regression	 	 
Sub Category: SVM	 	 
Sub Category: Kernel SVM	 	 
Sub Category: Ensemble Learning	 	 
			
Step 6- Model Evaluation
Sub Category: Confusion Matrix & Accuracy	 	 
Sub Category: AUC/ROC	 	 
Sub Category: Mean Squared Error	 	 
Sub Category: Recall, Precision and F1 Score	 	







# References:

Amira K; Ajith. A. (2013) ‘Modelling consumer loan default prediction using ensemble neural networks’ 2013 INTERNATIONAL CONFERENCE ON COMPUTING, ELECTRICAL AND ELECTRONIC ENGINEERING (ICCEEE) 

Arutjothi, G; Senthamarai, C. (2017) ‘Prediction of loan status in commercial bank using machine learning classifier’ 2017 International Conference on Intelligent Sustainable Systems (ICISS) 

Das, D; Sadiq, A; Ahmad, N; Lloret,J. (2017) ‘Stock Market Prediction with Big Data Through Hybridization of Data Mining and Optimized Neural Network Techniques.’ Journal of Multiple-Valued Logic & Soft Computing. Vol. 29 Issue 1/2, p157-181.

Hyeongjun, K; Hoon, C; Doojin, R. (2018) ‘An Empirical Study on Credit Card Loan Delinquency
‘Economic Systems, vol 42

Kumar, A; Garg, I; Kaur, S. (2018) ‘Loan Approval Prediction based on Machine Learning Approach’ IOSR Journal of Computer Engineering

Lendingclub (no date) Lending Club Statistics [Online] Available at: https://www.lendingclub.com/info/download-data.action [Accessed 14 Jan 2019]

Maroco, J; Silva, D; Rodrigues. A; Guerreiro, M; Santana, I; de Mendonça, A. (2011) ‘Data mining methods in the prediction of Dementia: A real-data comparison of the accuracy, sensitivity and specificity of linear discriminant analysis, logistic regression, neural networks, support vector machines, classification trees and random forests’ BMC Research Notes.

Thomas, L. (2000) ‘A survey of credit and behavioral scoring: forecasting financial risk of lending to consumers’ International Journal of Forecasting

Reynolds, A.; Richards, G; Rayward-Smith, V. (2004) ‘The Application of K-Medoids and PAM to the Clustering of Rules’ Intelligent Data Engineering and Automated Learning - IDEAL 2004 









