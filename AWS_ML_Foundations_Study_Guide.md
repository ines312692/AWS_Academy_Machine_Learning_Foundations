# AWS Machine Learning Foundations - Study Guide & QCM

## Table of Contents
1. [Module 2: Introducing Machine Learning](#module-2-introducing-machine-learning)
2. [Module 3: ML Pipeline with Amazon SageMaker](#module-3-ml-pipeline-with-amazon-sagemaker)
3. [Module 4: Forecasting](#module-4-forecasting)
4. [Module 5: Computer Vision](#module-5-computer-vision)
5. [Module 6: Natural Language Processing](#module-6-natural-language-processing)
6. [Module 7: Generative AI](#module-7-generative-ai)

---

## Module 2: Introducing Machine Learning

### Key Concepts Summary
- **Machine Learning**: Algorithms that learn from data to make predictions or decisions
- **Types of ML**: Supervised, Unsupervised, Reinforcement Learning
- **Supervised Learning**: Uses labeled data (classification, regression)
- **Unsupervised Learning**: Uses unlabeled data (clustering, dimensionality reduction)
- **Model Training**: Process of learning patterns from data
- **Overfitting**: Model performs well on training data but poorly on new data
- **Underfitting**: Model is too simple and performs poorly on all data

### QCM Questions

**Q1. What is Machine Learning?**
- A) Programming computers with explicit rules
- B) Teaching computers to learn from data without being explicitly programmed ✓
- C) Creating databases for storing information
- D) Designing user interfaces

**Q2. Which type of learning uses labeled data?**
- A) Unsupervised Learning
- B) Reinforcement Learning
- C) Supervised Learning ✓
- D) Transfer Learning

**Q3. What is an example of a classification problem?**
- A) Predicting house prices
- B) Detecting spam emails ✓
- C) Forecasting stock prices
- D) Measuring temperature

**Q4. What is overfitting?**
- A) Model is too simple
- B) Model memorizes training data and performs poorly on new data ✓
- C) Model performs well on all data
- D) Training data is insufficient

**Q5. Which is an example of unsupervised learning?**
- A) Email spam detection
- B) Customer segmentation/clustering ✓
- C) Image classification
- D) Sentiment analysis

**Q6. What is a regression problem?**
- A) Predicting categorical labels
- B) Predicting continuous numerical values ✓
- C) Grouping similar data points
- D) Detecting anomalies

**Q7. What is a feature in machine learning?**
- A) The predicted output
- B) An input variable used for prediction ✓
- C) The algorithm used
- D) The training dataset

**Q8. What is the purpose of a test dataset?**
- A) To train the model
- B) To tune hyperparameters
- C) To evaluate model performance on unseen data ✓
- D) To store the model

---

## Module 3: ML Pipeline with Amazon SageMaker

### Key Concepts Summary

#### 1. Formulating ML Problems
- Define business objectives
- Identify if ML is the right solution
- Choose problem type (classification, regression, clustering)

#### 2. Data Collection & Security
- **Data Sources**: S3, databases, streaming data
- **ETL**: Extract, Transform, Load
- **Security**: Encryption at rest and in transit, IAM policies, VPC

#### 3. Data Evaluation
- **Descriptive Statistics**: Mean, median, mode, standard deviation
- **Correlations**: Relationship between features
- **Data Quality**: Missing values, duplicates, inconsistencies

#### 4. Feature Engineering
- **Data Cleaning**: Handle missing values, outliers
- **Encoding**: Convert categorical data to numerical (one-hot encoding, label encoding)
- **Normalization/Standardization**: Scale features
- **Feature Selection**: Choose relevant features

#### 5. Training
- **Amazon SageMaker**: Managed service for training ML models
- **Built-in Algorithms**: XGBoost, Linear Learner, K-Means, etc.
- **Training Jobs**: Specify algorithm, data location, compute resources

#### 6. Model Deployment
- **Hosting**: Deploy model to an endpoint
- **Inference**: Make predictions on new data
- **Real-time vs Batch**: Synchronous vs asynchronous predictions

#### 7. Model Evaluation
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Confusion Matrix**: TP, TN, FP, FN
- **Threshold Selection**: Balance between precision and recall

#### 8. Hyperparameter Tuning
- **Hyperparameters**: Parameters set before training (learning rate, number of trees)
- **Tuning**: Find optimal hyperparameter values
- **SageMaker Autopilot**: Automated ML model building

### QCM Questions

**Q9. What is Amazon SageMaker?**
- A) A database service
- B) A fully managed machine learning service ✓
- C) A data storage service
- D) A visualization tool

**Q10. What does ETL stand for?**
- A) Evaluate, Test, Learn
- B) Extract, Transform, Load ✓
- C) Encode, Train, Label
- D) Execute, Transfer, Log

**Q11. Where does Amazon SageMaker typically store training data?**
- A) Amazon EC2
- B) Amazon S3 ✓
- C) Amazon RDS
- D) Local filesystem

**Q12. What is one-hot encoding used for?**
- A) Normalizing numerical data
- B) Converting categorical variables to numerical format ✓
- C) Removing outliers
- D) Splitting data into train/test sets

**Q13. What is the purpose of data normalization?**
- A) To remove duplicate data
- B) To scale features to a similar range ✓
- C) To encrypt data
- D) To compress data

**Q14. What does a confusion matrix show?**
- A) Model training time
- B) Actual vs predicted classifications ✓
- C) Feature importance
- D) Data distribution

**Q15. What is precision in classification?**
- A) TP / (TP + FP) - Proportion of positive predictions that were correct ✓
- B) TP / (TP + FN)
- C) (TP + TN) / Total
- D) TN / (TN + FP)

**Q16. What is recall (sensitivity)?**
- A) TP / (TP + FP)
- B) TP / (TP + FN) - Proportion of actual positives correctly identified ✓
- C) (TP + TN) / Total
- D) TN / (TN + FN)

**Q17. What is the F1-Score?**
- A) Average of precision and recall
- B) Harmonic mean of precision and recall ✓
- C) Product of precision and recall
- D) Difference between precision and recall

**Q18. What is a hyperparameter?**
- A) A parameter learned during training
- B) A parameter set before training that controls the learning process ✓
- C) The final model output
- D) A feature in the dataset

**Q19. What is Amazon SageMaker Autopilot?**
- A) A data cleaning service
- B) An automated machine learning service that builds and tunes models ✓
- C) A model deployment tool
- D) A data visualization service

**Q20. What is the purpose of a validation dataset?**
- A) To train the model
- B) To tune hyperparameters and prevent overfitting ✓
- C) To evaluate final model performance
- D) To store predictions

**Q21. What is feature selection?**
- A) Choosing the ML algorithm
- B) Identifying and selecting the most relevant features for the model ✓
- C) Normalizing data
- D) Splitting data into train/test sets

**Q22. How do you handle missing data?**
- A) Always delete rows with missing values
- B) Imputation (fill with mean/median/mode) or deletion ✓
- C) Ignore it during training
- D) Convert to zeros

**Q23. What is an outlier?**
- A) A normal data point
- B) A data point significantly different from other observations ✓
- C) A missing value
- D) A categorical variable

**Q24. What is the ROC curve used for?**
- A) Visualizing training loss
- B) Evaluating classification model performance across different thresholds ✓
- C) Showing data distribution
- D) Comparing different algorithms

**Q25. What does AUC stand for?**
- A) Area Under Curve (ROC curve) ✓
- B) Average Unified Classification
- C) Automated User Classification
- D) Accuracy Under Constraint

---

## Module 4: Forecasting

### Key Concepts Summary
- **Time Series Data**: Data points indexed in time order
- **Forecasting**: Predicting future values based on historical patterns
- **Components**: Trend, Seasonality, Cyclical patterns, Noise
- **Amazon Forecast**: Managed service for time series forecasting
- **Special Considerations**: Temporal dependencies, stationarity, seasonality
- **Amazon SageMaker Canvas**: No-code ML tool for forecasting

### QCM Questions

**Q26. What is time series data?**
- A) Data organized in alphabetical order
- B) Data points collected or recorded at successive time intervals ✓
- C) Static snapshot data
- D) Categorical data

**Q27. What is seasonality in time series?**
- A) Random fluctuations
- B) Long-term trend
- C) Regular patterns that repeat at fixed intervals ✓
- D) One-time events

**Q28. What is Amazon Forecast?**
- A) A weather prediction service
- B) A managed service for time series forecasting using ML ✓
- C) A financial planning tool
- D) A calendar application

**Q29. Which is NOT a component of time series data?**
- A) Trend
- B) Seasonality
- C) Correlation coefficient ✓
- D) Noise

**Q30. What is a key challenge with time series data?**
- A) Data is always numerical
- B) Temporal dependencies between observations ✓
- C) Too much data storage
- D) Lack of patterns

**Q31. What is Amazon SageMaker Canvas used for?**
- A) Drawing diagrams
- B) No-code machine learning and forecasting ✓
- C) Writing code
- D) Database management

**Q32. What does stationarity mean in time series?**
- A) Data doesn't change over time
- B) Statistical properties (mean, variance) are constant over time ✓
- C) Data is seasonal
- D) Data has a trend

**Q33. What is a common application of forecasting?**
- A) Image classification
- B) Demand prediction for inventory management ✓
- C) Text translation
- D) Face recognition

---

## Module 5: Computer Vision (CV)

### Key Concepts Summary
- **Computer Vision**: Field of AI that enables computers to interpret visual information
- **Image Classification**: Categorizing images into predefined classes
- **Object Detection**: Identifying and locating objects in images
- **Facial Recognition**: Identifying or verifying people from facial features
- **Amazon Rekognition**: Managed computer vision service
- **Image Analysis**: Detecting labels, scenes, activities, text, inappropriate content
- **Video Analysis**: Analyzing video streams for objects, people, activities
- **Amazon Ground Truth**: Data labeling service for creating training datasets
- **Training/Test Datasets**: Properly split and labeled image data

### QCM Questions

**Q34. What is Computer Vision?**
- A) Improving human eyesight
- B) Enabling computers to derive information from images and videos ✓
- C) Creating computer monitors
- D) Video editing software

**Q35. What is Amazon Rekognition?**
- A) A database service
- B) A managed computer vision service for image and video analysis ✓
- C) A speech recognition service
- D) A text processing service

**Q36. What can Amazon Rekognition detect in images?**
- A) Only faces
- B) Objects, scenes, text, faces, activities, inappropriate content ✓
- C) Only text
- D) Audio content

**Q37. What is facial recognition used for?**
- A) Image compression
- B) Identifying or verifying individuals based on facial features ✓
- C) Color detection
- D) Image resizing

**Q38. What is Amazon Ground Truth?**
- A) A GPS service
- B) A data labeling service for building accurate ML training datasets ✓
- C) A database service
- D) A server hosting service

**Q39. What is object detection?**
- A) Finding the oldest object
- B) Identifying and locating objects within an image with bounding boxes ✓
- C) Counting pixels
- D) Image compression

**Q40. Why is data labeling important in computer vision?**
- A) It makes images prettier
- B) It provides ground truth labels for supervised learning ✓
- C) It reduces file size
- D) It enhances image quality

**Q41. What is image classification?**
- A) Sorting images by file size
- B) Categorizing an entire image into predefined classes ✓
- C) Editing images
- D) Compressing images

**Q42. What is the difference between training and test datasets?**
- A) Training data is larger
- B) Training data is used to learn, test data evaluates performance on unseen data ✓
- C) Test data has more labels
- D) There is no difference

**Q43. What is a bounding box?**
- A) A type of database
- B) A rectangular box drawn around an object in an image ✓
- C) A storage container
- D) A security feature

**Q44. What is video analysis?**
- A) Editing videos
- B) Analyzing video content frame by frame to detect objects, activities, people ✓
- C) Compressing videos
- D) Converting video formats

---

## Module 6: Natural Language Processing (NLP)

### Key Concepts Summary
- **NLP**: Field of AI for understanding and processing human language
- **Text Analysis**: Extracting insights from text
- **Sentiment Analysis**: Determining emotional tone (positive, negative, neutral)
- **Entity Recognition**: Identifying named entities (people, places, organizations)
- **Key Phrase Extraction**: Identifying important phrases in text
- **Language Detection**: Identifying the language of text
- **Amazon Comprehend**: NLP service for text analysis
- **Amazon Polly**: Text-to-speech service
- **Amazon Translate**: Neural machine translation service
- **Amazon Lex**: Service for building conversational interfaces (chatbots)
- **Amazon Transcribe**: Speech-to-text service

### QCM Questions

**Q45. What is Natural Language Processing (NLP)?**
- A) Processing computer code
- B) Enabling computers to understand, interpret, and generate human language ✓
- C) Image processing
- D) Database queries

**Q46. What is Amazon Comprehend?**
- A) A translation service
- B) A natural language processing service for text analysis and insights ✓
- C) A speech service
- D) A chatbot service

**Q47. What is sentiment analysis?**
- A) Counting words
- B) Determining the emotional tone or opinion expressed in text ✓
- C) Translating text
- D) Correcting grammar

**Q48. What does Amazon Polly do?**
- A) Analyzes text sentiment
- B) Converts text to lifelike speech ✓
- C) Translates languages
- D) Recognizes speech

**Q49. What is Amazon Translate used for?**
- A) Text analysis
- B) Neural machine translation between languages ✓
- C) Speech recognition
- D) Image recognition

**Q50. What is Amazon Lex?**
- A) A text analysis service
- B) A service for building conversational interfaces using voice and text ✓
- C) A translation service
- D) A database service

**Q51. What is entity recognition in NLP?**
- A) Counting words
- B) Identifying and classifying named entities (people, places, organizations, dates) ✓
- C) Translating text
- D) Generating text

**Q52. What is key phrase extraction?**
- A) Deleting unnecessary text
- B) Identifying the most important phrases and concepts in text ✓
- C) Translating phrases
- D) Counting phrases

**Q53. What is Amazon Transcribe?**
- A) A translation service
- B) An automatic speech recognition service that converts speech to text ✓
- C) A text-to-speech service
- D) A chatbot service

**Q54. What is language detection?**
- A) Finding errors in code
- B) Automatically identifying the language of a given text ✓
- C) Translating languages
- D) Teaching languages

**Q55. What is a chatbot?**
- A) A computer virus
- B) A conversational interface that simulates human conversation ✓
- C) A database query tool
- D) An email client

**Q56. What is tokenization in NLP?**
- A) Encrypting text
- B) Breaking text into smaller units (words, phrases, sentences) ✓
- C) Translating text
- D) Analyzing sentiment

**Q57. What can Amazon Comprehend detect?**
- A) Only sentiment
- B) Sentiment, entities, key phrases, language, syntax, topics ✓
- C) Only languages
- D) Only grammar errors

---

## Module 7: Generative AI

### Key Concepts Summary
- **Generative AI**: AI systems that can create new content (text, images, code, audio)
- **Large Language Models (LLMs)**: Neural networks trained on massive text data
- **Foundation Models**: Large pre-trained models that can be adapted for various tasks
- **Prompt Engineering**: Crafting effective inputs to get desired outputs from AI
- **Amazon Bedrock**: Service for building generative AI applications with foundation models
- **Use Cases**: Content generation, code completion, summarization, question answering
- **Responsible AI**: Ethical considerations, bias, safety, transparency

### QCM Questions

**Q58. What is Generative AI?**
- A) AI that only analyzes data
- B) AI that creates new content (text, images, audio, code) ✓
- C) AI that only classifies data
- D) AI that stores data

**Q59. What is a Large Language Model (LLM)?**
- A) A small database
- B) A neural network trained on vast amounts of text data to understand and generate language ✓
- C) A translation dictionary
- D) A spell checker

**Q60. What is Amazon Bedrock?**
- A) A construction service
- B) A fully managed service for building generative AI applications with foundation models ✓
- C) A database service
- D) A storage service

**Q61. What is prompt engineering?**
- A) Building physical structures
- B) Designing effective inputs to guide AI model outputs ✓
- C) Programming databases
- D) Creating user interfaces

**Q62. What is a foundation model?**
- A) A simple algorithm
- B) A large pre-trained model that can be adapted for various downstream tasks ✓
- C) A database schema
- D) A programming language

**Q63. Which is an application of Generative AI?**
- A) Only email classification
- B) Content creation, code generation, summarization, creative writing ✓
- C) Only data storage
- D) Only database queries

**Q64. What is an important consideration for Generative AI?**
- A) Only speed
- B) Responsible AI practices including bias, safety, and transparency ✓
- C) Only cost
- D) Only data size

**Q65. What can generative AI models create?**
- A) Only text
- B) Text, images, audio, video, code, and more ✓
- C) Only images
- D) Only databases

**Q66. What is fine-tuning in the context of foundation models?**
- A) Deleting data
- B) Adapting a pre-trained model to a specific task with additional training ✓
- C) Compressing models
- D) Installing software

**Q67. What is hallucination in AI?**
- A) A visualization technique
- B) When AI generates false or nonsensical information presented as fact ✓
- C) A type of neural network
- D) A data processing method

---

## General AWS & ML Concepts

### QCM Questions

**Q68. What is Amazon S3?**
- A) A database service
- B) Scalable object storage service for data, backups, and archives ✓
- C) A computing service
- D) A machine learning service

**Q69. What does IAM stand for?**
- A) Internet Access Management
- B) Identity and Access Management ✓
- C) Image Analysis Method
- D) Intelligent Algorithm Model

**Q70. What is the AWS Management Console?**
- A) A physical device
- B) A web-based interface for accessing and managing AWS services ✓
- C) A programming language
- D) A database

**Q71. What is the purpose of splitting data into training and test sets?**
- A) To save storage space
- B) To evaluate model performance on unseen data and prevent overfitting ✓
- C) To make training faster
- D) To compress data

**Q72. What is cross-validation?**
- A) Checking data twice
- B) A technique to assess model performance by splitting data into multiple folds ✓
- C) Translating data
- D) Encrypting data

**Q73. What is bias in machine learning?**
- A) A model parameter
- B) Systematic error or unfairness in model predictions ✓
- C) A training algorithm
- D) A data format

**Q74. What is the difference between AI, ML, and Deep Learning?**
- A) They are the same
- B) AI is the broad field, ML is a subset using algorithms, Deep Learning uses neural networks ✓
- C) ML is broader than AI
- D) Deep Learning doesn't use algorithms

**Q75. What is a neural network?**
- A) A computer network
- B) A computing system inspired by biological neural networks in the brain ✓
- C) A database structure
- D) A cloud service

**Q76. What is batch processing?**
- A) Processing data one record at a time
- B) Processing large volumes of data in groups ✓
- C) Real-time processing
- D) Manual processing

**Q77. What is real-time inference?**
- A) Batch predictions
- B) Making predictions immediately as data arrives ✓
- C) Historical analysis
- D) Offline processing

**Q78. What is model versioning?**
- A) Deleting old models
- B) Tracking different versions of a model over time ✓
- C) Compressing models
- D) Encrypting models

**Q79. What is feature importance?**
- A) The size of features
- B) A measure of how much each feature contributes to model predictions ✓
- C) The order of features
- D) The type of features

**Q80. What is transfer learning?**
- A) Transferring data between servers
- B) Using knowledge from a pre-trained model for a new but related task ✓
- C) Moving models to production
- D) Translating languages

---

## Practice Test (Mixed Questions)

**Q81. Which AWS service would you use for automated data labeling?**
- A) Amazon S3
- B) Amazon Ground Truth ✓
- C) Amazon EC2
- D) Amazon RDS

**Q82. What metric would you prioritize for a cancer detection model?**
- A) Accuracy only
- B) Recall (to minimize false negatives and catch all cancer cases) ✓
- C) Precision only
- D) Training speed

**Q83. What is the purpose of normalization?**
- A) To remove data
- B) To scale numerical features to a similar range (e.g., 0-1) ✓
- C) To classify data
- D) To encrypt data

**Q84. Which service provides pre-trained models for vision and NLP?**
- A) Amazon S3
- B) Amazon Rekognition and Amazon Comprehend ✓
- C) Amazon EC2
- D) Amazon VPC

**Q85. What is the k in k-fold cross-validation?**
- A) Number of features
- B) Number of splits/folds in the dataset ✓
- C) Number of models
- D) Number of predictions

**Q86. What is ensemble learning?**
- A) Training one model
- B) Combining multiple models to improve performance ✓
- C) Deleting poor models
- D) Compressing models

**Q87. What is the curse of dimensionality?**
- A) Too little data
- B) Problems that arise when working with high-dimensional data ✓
- C) Slow computers
- D) Network issues

**Q88. What is data augmentation?**
- A) Deleting data
- B) Creating additional training data through transformations (rotation, flipping, etc.) ✓
- C) Compressing data
- D) Encrypting data

**Q89. What is a confusion matrix element TP (True Positive)?**
- A) Incorrectly predicted as positive
- B) Correctly predicted as positive ✓
- C) Correctly predicted as negative
- D) Incorrectly predicted as negative

**Q90. What does GPU acceleration help with?**
- A) Data storage
- B) Faster training of deep learning models ✓
- C) Network speed
- D) Data encryption

**Q91. What is the purpose of dropout in neural networks?**
- A) To increase model complexity
- B) To prevent overfitting by randomly dropping neurons during training ✓
- C) To speed up training
- D) To reduce data size

**Q92. What is a baseline model?**
- A) The final production model
- B) A simple model used as a reference point for comparison ✓
- C) The most complex model
- D) A visualization tool

**Q93. What is model drift?**
- A) Moving models to different servers
- B) Degradation of model performance over time as data patterns change ✓
- C) Training models faster
- D) Compressing models

**Q94. What is the difference between parameters and hyperparameters?**
- A) No difference
- B) Parameters are learned during training; hyperparameters are set before training ✓
- C) Hyperparameters are learned during training
- D) Parameters are always fixed

**Q95. What is batch size in training?**
- A) The size of the dataset
- B) The number of samples processed before updating model weights ✓
- C) The number of features
- D) The number of epochs

**Q96. What is an epoch in training?**
- A) A single data sample
- B) One complete pass through the entire training dataset ✓
- C) A model parameter
- D) A validation step

**Q97. What is learning rate?**
- A) How fast the model runs
- B) How much the model weights are adjusted during training ✓
- C) The amount of data processed
- D) The model accuracy

**Q98. What is regularization?**
- A) Making data regular
- B) Techniques to prevent overfitting by adding penalties to model complexity ✓
- C) Speeding up training
- D) Data cleaning

**Q99. What is A/B testing in ML?**
- A) Testing two features
- B) Comparing two models or versions in production to determine which performs better ✓
- C) Training two models simultaneously
- D) Using two datasets

**Q100. What is the purpose of a validation dataset?**
- A) To train the model
- B) To tune hyperparameters and monitor training without touching test data ✓
- C) To deploy the model
- D) To store predictions

---

## Answer Key Summary

### Module 2 (Q1-Q8)
1. B | 2. C | 3. B | 4. B | 5. B | 6. B | 7. B | 8. C

### Module 3 (Q9-Q25)
9. B | 10. B | 11. B | 12. B | 13. B | 14. B | 15. A | 16. B | 17. B | 18. B | 19. B | 20. B | 21. B | 22. B | 23. B | 24. B | 25. A

### Module 4 (Q26-Q33)
26. B | 27. C | 28. B | 29. C | 30. B | 31. B | 32. B | 33. B

### Module 5 (Q34-Q44)
34. B | 35. B | 36. B | 37. B | 38. B | 39. B | 40. B | 41. B | 42. B | 43. B | 44. B

### Module 6 (Q45-Q57)
45. B | 46. B | 47. B | 48. B | 49. B | 50. B | 51. B | 52. B | 53. B | 54. B | 55. B | 56. B | 57. B

### Module 7 (Q58-Q67)
58. B | 59. B | 60. B | 61. B | 62. B | 63. B | 64. B | 65. B | 66. B | 67. B

### General & Practice (Q68-Q100)
68. B | 69. B | 70. B | 71. B | 72. B | 73. B | 74. B | 75. B | 76. B | 77. B | 78. B | 79. B | 80. B | 81. B | 82. B | 83. B | 84. B | 85. B | 86. B | 87. B | 88. B | 89. B | 90. B | 91. B | 92. B | 93. B | 94. B | 95. B | 96. B | 97. B | 98. B | 99. B | 100. B

---

## Quick Study Tips

### For Your Test:
1. **Focus on AWS Services**: Know what each service does (SageMaker, Rekognition, Comprehend, Forecast, Polly, Translate, Lex, Bedrock)
2. **ML Pipeline Steps**: Understand the complete workflow from data collection to deployment
3. **Evaluation Metrics**: Master precision, recall, F1-score, accuracy, confusion matrix
4. **Data Preprocessing**: Know encoding, normalization, handling missing values
5. **Key Concepts**: Overfitting, hyperparameters, feature engineering, cross-validation
6. **Service Use Cases**: Match AWS services to appropriate use cases

### Memory Aids:
- **PRECISION**: "Of all I predicted positive, how many were correct?"
- **RECALL**: "Of all actual positives, how many did I find?"
- **F1-Score**: Balance between precision and recall
- **ETL**: Extract → Transform → Load
- **Training vs Test**: Train learns, Test evaluates unseen data

Good luck with your test! 🚀

---

## Additional Resources
- AWS Documentation: https://docs.aws.amazon.com
- AWS Skill Builder: https://skillbuilder.aws
- SageMaker Examples: https://github.com/aws/amazon-sagemaker-examples
