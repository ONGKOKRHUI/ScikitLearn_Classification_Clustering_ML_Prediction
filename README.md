# Student Performance and Travel Time Analysis

## Overview

This project showcases a comprehensive data analysis workflow, from data cleaning and preparation to the application of both supervised and unsupervised machine learning models. The analysis is divided into two main parts:

1.  **Student Grade Prediction:** A supervised learning task to predict student grade classifications based on various academic and demographic factors. This part demonstrates skills in data wrangling, feature scaling, model training (using Support Vector Machine and Decision Tree classifiers), and performance evaluation.
2.  **Travel Time Clustering:** An unsupervised learning task that involves selecting an external dataset, cleaning it, and applying K-Means clustering to identify distinct patterns in travel behavior based on distance and speed.

This project was completed as part of the FIT1043 unit and serves as a practical demonstration of fundamental data science and machine learning principles using Python and popular libraries like Pandas, Scikit-learn, and Matplotlib.

---

## Table of Contents

- [Key Features & Skills Demonstrated](#key-features--skills-demonstrated)
- [Project Breakdown](#project-breakdown)
  - [Part 1: Student Grade Prediction (Supervised Learning)](#part-1-student-grade-prediction-supervised-learning)
    - [A1: Data Wrangling](#a1-data-wrangling)
    - [A2: Supervised Learning Setup](#a2-supervised-learning-setup)
    - [A3: Classification Model Training](#a3-classification-model-training)
    - [A4: Prediction and Evaluation](#a4-prediction-and-evaluation)
    - [A5: Independent Evaluation on New Data](#a5-independent-evaluation-on-new-data)
  - [Part 2: Travel Time Analysis (Unsupervised Learning)](#part-2-travel-time-analysis-unsupervised-learning)
    - [B1: Dataset Selection and K-Means Clustering](#b1-dataset-selection-and-k-means-clustering)
- [Technologies Used](#technologies-used)
- [How to Run This Project](#how-to-run-this-project)

---

## Key Features & Skills Demonstrated

* **Data Wrangling & Cleaning:** Handling missing values (imputation with median), correcting data inconsistencies, identifying and removing outliers, and transforming data types.
* **Exploratory Data Analysis (EDA):** Identifying data quality issues and inconsistencies between related features (e.g., GPA and GradeClass).
* **Supervised Machine Learning:**
    * **Classification:** Building models to predict categorical outcomes (student grades).
    * **Model Implementation:** Training and evaluating **Support Vector Machine (SVM)** and **Decision Tree** classifiers.
    * **Model Evaluation:** Using confusion matrices and accuracy scores to compare model performance and select the best-performing model.
* **Feature Engineering & Scaling:** Selecting relevant features, dropping unnecessary ones, and applying `StandardScaler` to normalize data for optimal model performance.
* **Unsupervised Machine Learning:**
    * **Clustering:** Applying the **K-Means** algorithm to segment data into distinct groups without pre-existing labels.
* **Data Visualization:** Using `Matplotlib` to visualize the results of the K-Means clustering, making the identified patterns easy to interpret.
* **Python Programming:** Utilizing core Python libraries for a complete data analysis project.

---

## Project Breakdown

### Part 1: Student Grade Prediction (Supervised Learning)

This section focuses on building a model to predict a student's final grade classification.

#### A1: Data Wrangling

The initial dataset (`Student_List_A2.csv`) required several cleaning steps to ensure data quality and integrity:
* **Categorical Replacement:** Numerical grade classifications (0-4) in the `GradeClass` column were mapped to their corresponding letter grades (A-F) for better readability.
* **Missing Value Imputation:** Missing values in the `StudyTimeWeekly` column were identified and filled using the column's median value.
* **Outlier and Error Handling:** The `Absences` column was inspected for data quality issues. Negative values were identified as invalid data and removed. Outliers were detected using the IQR method and filtered out to prevent them from skewing the model.
* **Data Consistency Check:** A logical inconsistency between `GPA` and `GradeClass` was found. A function was created to derive the correct grade from the GPA, and the `GradeClass` column was updated accordingly, ensuring the data was consistent and accurate.

#### A2: Supervised Learning Setup

The cleaned data was prepared for model training:
* **Feature and Label Separation:** The dataset was split into features (X) and the target label (Y). `StudentID` and `GPA` were excluded as features because they are either an identifier or directly used to derive the target label.
* **Train-Test Split:** The data was partitioned into an 80% training set and a 20% testing set using `train_test_split` to allow for unbiased model evaluation.

#### A3: Classification Model Training

Two classification algorithms were trained on the prepared data:

1.  **Data Scaling:** Before training, the feature sets (`X_train` and `X_test`) were normalized using `StandardScaler`. This is crucial for distance-based algorithms like SVM to ensure that all features contribute equally to the result.
2.  **Support Vector Machine (SVM):** An SVM model with a linear kernel was implemented. This algorithm works by finding an optimal hyperplane that best separates the different classes in the feature space.
3.  **Decision Tree Classifier:** A second model, a Decision Tree, was trained using the 'entropy' criterion. This algorithm creates a tree-like model of decisions to classify the data.

#### A4: Prediction and Evaluation

Both models were used to make predictions on the unseen test data. Their performances were then compared:
* **Confusion Matrix:** A 5x5 confusion matrix was generated for each model to visualize the accuracy of predictions across all five grade classes.
* **Accuracy Comparison:** The overall accuracy of each model was calculated. The **SVM model achieved a higher accuracy (76.0%)** compared to the Decision Tree model (71.4%), indicating it was the better-performing model for this dataset.

#### A5: Independent Evaluation on New Data

The superior SVM model was used to predict the `GradeClass` for a new, unlabeled dataset (`Student_List_A2_Submission.csv`). The final predictions were then exported to a new CSV file.

### Part 2: Travel Time Analysis (Unsupervised Learning)

This section explores a different dataset to demonstrate skills in unsupervised learning.

#### B1: Dataset Selection and K-Means Clustering

* **Dataset:** The "Travel Times" dataset from openmv.net was chosen, which contained records of commutes with features like distance, speed, and time.
* **Data Wrangling:** The dataset was first inspected for missing values. The `FuelEconomy` and `Comments` columns contained nulls. As the number of affected rows was small, they were simply dropped to create a clean dataset for clustering.
* **K-Means Clustering:** The K-Means algorithm was applied to segment the data into **two clusters (k=2)** based on the `Distance` and `AvgMovingSpeed` features.
* **Visualization and Interpretation:** The resulting clusters were visualized using a scatter plot. The analysis revealed two distinct travel patterns:
    * **Cluster 1 (Purple):** Shorter distances with lower average moving speeds, likely representing travel on town roads with more traffic and lower speed limits.
    * **Cluster 2 (Yellow):** Longer distances with higher average moving speeds, characteristic of highway commutes.

---

## Technologies Used

* **Python**
* **Pandas:** For data manipulation, cleaning, and analysis.
* **Scikit-learn (sklearn):** For implementing machine learning models (SVM, Decision Tree, K-Means), data splitting, and performance evaluation.
* **Matplotlib:** For data visualization and plotting the clustering results.
* **Jupyter Notebook:** As the environment for interactive coding and analysis.

---

## How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [your-repository-url]
    cd [repository-name]
    ```

2.  **Ensure you have Python and Jupyter Notebook installed.**

3.  **Install the required libraries.** You can install them using pip:
    ```bash
    pip install pandas scikit-learn matplotlib jupyterlab
    ```

4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Or if you have JupyterLab:
    ```bash
    jupyter lab
    ```

5.  **Open the `FIT1043_A2.ipynb` file** and run the cells sequentially to reproduce the analysis. Make sure the `.csv` data files are in the same directory as the notebook.