# AI Projects at AIFFEL

This section provides an overview of key AI projects completed, highlighting the machine learning models, data preprocessing techniques, and optimization methods used throughout the learning process.

## Projects Overview Table

| Name                        | Dataset                                                                                 | Model & Algorithm                        | Optimization               | Result                                         | Libraries                                         |
|-----------------------------|-----------------------------------------------------------------------------------------|------------------------------------------|----------------------------|------------------------------------------------|---------------------------------------------------|
| Bike Sharing Demand Regression | [Bike Sharing](https://www.kaggle.com/code/gauravduttakiit/bike-sharing-multiple-linear-regression) | Gradient Boosting Regressor, <br> RandomForestClassifier             | IQR, Skewness and Kurtosis         | R2(score_train: 0.99, score_val: 0.95)                        | numpy, pandas, matplotlib, seaborn, scipy         |
| Diabetes Regression          | [diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)   | Linear Regression                                               | Learning Rate                     | MSE                                                           | numpy, pandas, sklearn, matplotlib, seaborn       |
| Tip Regression               | [tips]()                                                                                            | Linear Regression                                               | Learning Rate                     | MSE                                                           | numpy, pandas, sklearn, matplotlib, seaborn       |
| Data Visualization           | [Amazon stock data](https://finance.yahoo.com/quote/AMZN/history/?p=AMZN)                           | -                                                              | -                                 | Bar graph, Line graph, Scatter plot, Histogram, Heatmap        | matplotlib, pandas, datetime, numpy, seaborn      |
| Evaluation Metric            | [Iris](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)     | svm                                                            | -                                 | Precision, Recall, F-score                                    | sklearn(svm, roc_curve, auc)                      |
| Preprocessing                | [Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales)                         | -                                                              | -                                 | Missing data, Outlier, Normalization, Duplicates, One-hot encoding, Binning | os, pandas, numpy                                |
| Regularization               | [Fashion MNIST](https://keras.io/api/datasets/fashion_mnist/)                                       | -                                                              | -                                 | L1 Regularization, L2 Regularization, Lp norm, Dropout, Batch Normalization | sklearn, pandas, numpy                            |
| Scikit Learn                 | [wine](https://www.kaggle.com/code/cristianlapenta/wine-dataset-sklearn-machine-learning-project)   | Linear Regression, <br> RandomForestClassifier(+Other ML models)     | -                                 | Classification, Regression, Clustering, Dimensionality reduction | sklearn, pandas                                   |
| Unsupervised Learning        | [mnist](https://www.openml.org/search?type=data&status=active&id=554&sort=runs)                     | K-means, DBSCAN, PCA, T-SNE                                     | -                                 | -                                                             | sklearn, pandas, os, numpy                        |
| Pokemon                            | [Pokemon Dataset](https://www.kaggle.com/rounakbanik/pokemon)                                  | Exploratory Data Analysis (EDA)                           | Feature Engineering, Data Cleaning       | Statistical insights, visualization of Pokemon attributes   | pandas, numpy, matplotlib, seaborn              |
| Workflow and Model in Keras        | Custom synthetic dataset                                                                       | Keras Sequential & Functional API                        | Hyperparameter tuning                     | Built and analyzed deep learning models in Keras            | tensorflow, keras                               |
| 2019 2nd ML month with KaKR        | [Kaggle 2019 ML Competition](https://www.kaggle.com/c/2019-2nd-ml-month-with-kakr)            | Various ML models (XGBoost, <br> LightGBM, Neural Networks)    | Feature Engineering, Hyperparameter Tuning | Competitive leaderboard submission                      | sklearn, pandas, numpy, XGBoost, LightGBM       |
| Convolution 2D                     | [MNIST](https://keras.io/api/datasets/mnist/)                                                 | Convolutional Neural Network (CNN)                        | Data Augmentation, Dropout                 | Achieved high accuracy in digit classification            | tensorflow, keras, matplotlib                   |
| Camera Sticker                     | Custom Image Dataset                                                                          | Convolutional Neural Network (CNN)                        | Transfer Learning (MobileNetV2)            | Real-time camera filters using deep learning              | tensorflow, keras, OpenCV                        |
| Human Segmentation                    | Custom image dataset | U-Net (CNN-based) | Data augmentation, Transfer Learning | Accurate human segmentation from images | TensorFlow, Keras, OpenCV |
| Keras Tuner                           | Fashion MNIST | Convolutional Neural Network (CNN) | Hyperparameter tuning with Keras Tuner | Optimized CNN model achieving improved accuracy | TensorFlow, Keras, Keras Tuner |
| Sentiment Classification               | [Naver Movie Review](https://github.com/e9t/nsmc)                        | LSTM, GRU, KoBERT                               | Tokenization, Embedding, Fine-tuning | High accuracy in sentiment classification    | TensorFlow, PyTorch, Transformers, KoBERT |
| Transformer Chatbot                     | Custom Conversation Dataset                                             | Transformer-based Model (Seq2Seq, BERT)         | Preprocessing, Fine-tuning         | Natural language conversation chatbot        | TensorFlow, Transformers, Hugging Face     |
| Chatbot GPT                           | Custom conversation dataset | Transformer-based model (GPT-2) | Fine-tuning, Preprocessing | Interactive chatbot capable of generating natural responses | TensorFlow, Transformers, Hugging Face |
| Korean Conversation Type Classification | Custom Korean Dialogue Dataset                                          | LSTM, GRU, KoBERT                              | Data Augmentation, Embedding Tuning | Accurate classification of conversation types | TensorFlow, PyTorch, KoBERT, Scikit-learn  |
| Topic Modeling                        |         |                   |              |        |           |
| Reuters Classification                |         |                   |              |        |           |
| WEAT                                  |         |                   |              |        |           |
| Word Embedding                        |         |                   |              |        |           |




## Key Projects and Skills Acquired

### 1. Bike Sharing Demand Regression  
- **Objective**: Predict bike rental demand using regression models.  
- **Techniques**: Applied Gradient Boosting Regressor and RandomForestClassifier. Optimized data distribution using IQR, Skewness, and Kurtosis.  

### 2. Diabetes Regression  
- **Objective**: Predict diabetes progression using regression analysis.  
- **Techniques**: Built a Linear Regression model and evaluated performance using Mean Squared Error (MSE).  

### 3. Tip Regression  
- **Objective**: Predict restaurant tip amounts.  
- **Techniques**: Used Linear Regression and visualized data to analyze patterns.  

### 4. Data Visualization (Amazon Stock Data)  
- **Objective**: Visualize Amazon stock data for time series analysis.  
- **Techniques**: Created bar graphs, line graphs, scatter plots, histograms, and heatmaps.  

### 5. Evaluation Metric (Iris Dataset)  
- **Objective**: Classify the Iris dataset and evaluate model performance.  
- **Techniques**: Used Support Vector Machine (SVM) and evaluated results with Precision, Recall, and F-score.  

### 6. Preprocessing (Video Game Sales)  
- **Objective**: Clean and preprocess the video game sales dataset.  
- **Techniques**: Handled missing data, outliers, normalization, and one-hot encoding.  

### 7. Regularization (Fashion MNIST)  
- **Objective**: Prevent overfitting in deep learning models.  
- **Techniques**: Applied L1 and L2 regularization, dropout, and batch normalization.  

### 8. Scikit-Learn (Wine Dataset)  
- **Objective**: Implement various ML models for classification, regression, and clustering.  
- **Techniques**: Used Linear Regression, RandomForestClassifier, and other Scikit-Learn models.  

### 9. Unsupervised Learning (MNIST Dataset)  
- **Objective**: Perform clustering and dimensionality reduction on handwritten digits.  
- **Techniques**: Applied K-means, DBSCAN, PCA, and T-SNE for feature extraction.  

### 10. Pokemon EDA  
- **Objective**: Conduct Exploratory Data Analysis (EDA) on the Pokemon dataset.  
- **Techniques**: Cleaned data, performed statistical analysis, and visualized attributes.  

### 11. Workflow and Model in Keras  
- **Objective**: Build and optimize deep learning models using Keras.  
- **Techniques**: Implemented Sequential and Functional API architectures.  

### 12. 2019 2nd ML Month with KaKR  
- **Objective**: Compete in Kaggle’s ML competition and improve leaderboard ranking.  
- **Techniques**: Used XGBoost, LightGBM, and Neural Networks with feature engineering and hyperparameter tuning.  

### 13. Convolution 2D  
- **Objective**: Develop a CNN for digit classification using MNIST.  
- **Techniques**: Applied Conv2D, MaxPooling, Dense layers, and data augmentation.  

### 14. Camera Sticker  
- **Objective**: Create an AI-based camera filter system.  
- **Techniques**: Used a CNN with MobileNetV2 for real-time image and video processing.

### 15. Human Segmentation
- **Objective**: Develop a deep learning model for human segmentation.
- **Techniques**: Implemented U-Net, utilized data augmentation for improved performance, and applied transfer learning.

### 16. Keras Tuner
- **Objective**: Optimize hyperparameters for CNN models.
- **Techniques**: Used Keras Tuner to explore optimal learning rates, number of filters, and dropout rates.

### 17. Sentiment Classification  
- **Objective**: Classify movie reviews as positive or negative.  
- **Techniques**: Used LSTM, GRU, and KoBERT for sentiment analysis. Applied tokenization, embedding, and fine-tuning strategies.  

### 18. Transformer Chatbot  
- **Objective**: Develop a chatbot capable of generating natural conversations.  
- **Techniques**: Implemented Transformer-based sequence-to-sequence and BERT models. Applied preprocessing and fine-tuning techniques.  

### 19. Chatbot GPT
- **Objective**: Develop a conversational AI model using GPT.
- **Techniques**: Fine-tuned a GPT-2 model on a custom dataset, applied preprocessing techniques for better responses.
- **Result**: Created an interactive chatbot capable of generating coherent and contextually relevant responses.

### 20. Korean Conversation Type Classification  
- **Objective**: Classify types of conversations in Korean dialogue datasets.  
- **Techniques**: Used LSTM, GRU, and KoBERT for classification. Optimized performance using data augmentation and embedding tuning.  

