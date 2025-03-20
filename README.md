# AI Projects at AIFFEL

This section provides an overview of key AI projects completed, highlighting the machine learning models, data preprocessing techniques, and optimization methods used throughout the learning process.

## Projects Overview Table

| Name                        | Dataset                                                                                             | Model & Algorithm                                              | Optimization                      | Result                                                        | Libraries                                          |
|-----------------------------|-----------------------------------------------------------------------------------------|------------------------------------------------------|-----------------------------------|---------------------------------------------------------------|---------------------------------------------------|
| Bike Sharing Demand Regression | [Bike Sharing](https://www.kaggle.com/code/gauravduttakiit/bike-sharing-multiple-linear-regression) | Gradient Boosting Regressor, RandomForestClassifier             | IQR, Skewness and Kurtosis         | R2(score_train: 0.99, score_val: 0.95)                        | numpy, pandas, matplotlib, seaborn, scipy         |
| Diabetes Regression          | [diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)   | Linear Regression                                               | Learning Rate                     | MSE                                                           | numpy, pandas, sklearn, matplotlib, seaborn       |
| Tip Regression               | [tips]()                                                                                            | Linear Regression                                               | Learning Rate                     | MSE                                                           | numpy, pandas, sklearn, matplotlib, seaborn       |
| Data Visualization           | [Amazon stock data](https://finance.yahoo.com/quote/AMZN/history/?p=AMZN)                           | -                                                              | -                                 | Bar graph, Line graph, Scatter plot, Histogram, Heatmap        | matplotlib, pandas, datetime, numpy, seaborn      |
| Evaluation Metric            | [Iris](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)     | svm                                                            | -                                 | Precision, Recall, F-score                                    | sklearn(svm, roc_curve, auc)                      |
| Preprocessing                | [Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales)                         | -                                                              | -                                 | Missing data, Outlier, Normalization, Duplicates, One-hot encoding, Binning | os, pandas, numpy                                |
| Regularization               | [Fashion MNIST](https://keras.io/api/datasets/fashion_mnist/)                                       | -                                                              | -                                 | L1 Regularization, L2 Regularization, Lp norm, Dropout, Batch Normalization | sklearn, pandas, numpy                            |
| Scikit Learn                 | [wine](https://www.kaggle.com/code/cristianlapenta/wine-dataset-sklearn-machine-learning-project)   | Linear Regression, RandomForestClassifier(+Other ML models)     | -                                 | Classification, Regression, Clustering, Dimensionality reduction | sklearn, pandas                                   |
| Unsupervised Learning        | [mnist](https://www.openml.org/search?type=data&status=active&id=554&sort=runs)                     | K-means, DBSCAN, PCA, T-SNE                                     | -                                 | -                                                             | sklearn, pandas, os, numpy                        |
| Pokemon                            | [Pokemon Dataset](https://www.kaggle.com/rounakbanik/pokemon)                                  | Exploratory Data Analysis (EDA)                           | Feature Engineering, Data Cleaning       | Statistical insights, visualization of Pokemon attributes   | pandas, numpy, matplotlib, seaborn              |
| Workflow and Model in Keras        | Custom synthetic dataset                                                                       | Keras Sequential & Functional API                        | Hyperparameter tuning                     | Built and analyzed deep learning models in Keras            | tensorflow, keras                               |
| 2019 2nd ML month with KaKR        | [Kaggle 2019 ML Competition](https://www.kaggle.com/c/2019-2nd-ml-month-with-kakr)            | Various ML models (XGBoost, LightGBM, Neural Networks)    | Feature Engineering, Hyperparameter Tuning | Competitive leaderboard submission                      | sklearn, pandas, numpy, XGBoost, LightGBM       |
| Convolution 2D                     | [MNIST](https://keras.io/api/datasets/mnist/)                                                 | Convolutional Neural Network (CNN)                        | Data Augmentation, Dropout                 | Achieved high accuracy in digit classification            | tensorflow, keras, matplotlib                   |
| Camera Sticker                     | Custom Image Dataset                                                                          | Convolutional Neural Network (CNN)                        | Transfer Learning (MobileNetV2)            | Real-time camera filters using deep learning              | tensorflow, keras, OpenCV                        |
| Human Segmentation                    |         |                   |              |        |           |
| Keras Tuner                           |         |                   |              |        |           |
| Sentiment Classification              |         |                   |              |        |           |
| Transformer Chatbot                   |         |                   |              |        |           |
| Chatbot GPT                           |         |                   |              |        |           |
| Korean Conversation Type Classification |       |                   |              |        |           |
| Topic Modeling                        |         |                   |              |        |           |
| Reuters Classification                |         |                   |              |        |           |
| WEAT                                  |         |                   |              |        |           |
| Word Embedding                        |         |                   |              |        |           |




## Key Projects and Skills Acquired


1. **Bike Sharing Demand Regression**
   - **Objective**: Predict bike rental demand using regression models.
   - **Techniques**: Implemented Gradient Boosting Regressor and RandomForestClassifier. Optimized data distribution using IQR, Skewness, and Kurtosis.

2. **Diabetes Regression**
   - **Objective**: Predict diabetes progression using regression analysis.
   - **Techniques**: Built a linear regression model and evaluated its performance using Mean Squared Error (MSE).

3. **Tip Regression**
   - **Objective**: Predict restaurant tip amounts.
   - **Techniques**: Applied Linear Regression and visualized data to better understand relationships and patterns.

4. **Data Visualization (Amazon stock data)**
   - **Objective**: Visualize Amazon stock data for time series analysis.
   - **Techniques**: Created bar graphs, line graphs, scatter plots, histograms, and heatmaps for data interpretation.

5. **Evaluation Metric (Iris dataset)**
   - **Objective**: Perform classification on the Iris dataset and evaluate model performance.
   - **Techniques**: Used Support Vector Machine (SVM) and evaluated results with metrics such as Precision, Recall, and F-score.

6. **Preprocessing (Video Game Sales)**
   - **Objective**: Clean and preprocess the video game sales dataset.
   - **Techniques**: Managed missing data, detected outliers, and applied normalization and one-hot encoding to prepare data for analysis.

7. **Regularization (Fashion MNIST)**
   - **Objective**: Prevent overfitting on the Fashion MNIST dataset.
   - **Techniques**: Applied L1 and L2 regularization, dropout, and batch normalization to enhance model generalization.

8. **Scikit Learn (wine dataset)**
   - **Objective**: Implement multiple machine learning models for classification, regression, and dimensionality reduction.
   - **Techniques**: Utilized Linear Regression, RandomForestClassifier, and other models available in Scikit Learn.

9. **Unsupervised Learning (MNIST dataset)**
   - **Objective**: Perform clustering and dimensionality reduction on the MNIST dataset.
   - **Techniques**: Applied K-means, DBSCAN, PCA, and T-SNE for unsupervised learning and visualization.

10. **Pokemon EDA**  
- **Objective**: Perform Exploratory Data Analysis (EDA) on the Pokemon dataset.  
- **Techniques**: Cleaned data, performed statistical analysis, and visualized various attributes of Pokemon to gain insights.  

11. **Workflow and Model in Keras**  
- **Objective**: Learn how to build and structure models using Keras Sequential and Functional API.  
- **Techniques**: Implemented various model architectures and compared their performance.  

12. **2019 2nd ML Month with KaKR**  
- **Objective**: Participate in Kaggle’s ML competition and improve ranking on the leaderboard.  
- **Techniques**: Implemented multiple ML models, performed feature engineering, and fine-tuned hyperparameters for optimal performance.  

13. **Convolution 2D**  
- **Objective**: Build a CNN model for image classification using the MNIST dataset.  
- **Techniques**: Designed a deep learning architecture using Conv2D, MaxPooling, and Dense layers, and applied data augmentation techniques to improve generalization.  

14. **Camera Sticker**  
- **Objective**: Develop an AI-based camera filter system using deep learning.  
- **Techniques**: Implemented a CNN model with transfer learning using MobileNetV2 to apply real-time stickers to images and videos.  
