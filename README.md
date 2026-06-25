# Smart-Notification-Prioritizer---NotifAI-
NotifAI is an ML-powered notification intelligence system that prioritizes notifications using machine learning and user behavior analysis. It analyzes app usage, interaction patterns, and notification activity to rank alerts by importance, creating a smarter and more personalized mobile notification experience.

--------------------------------------------------------------------------------------------------------------------------------------
Problem Statement

Modern users receive a large number of notifications daily, making it difficult to identify important ones. NotifAI solves this by using machine learning models to classify and prioritize notifications in real time.

----------------------------------------------------------------------------------------------------------------------------------------
Solution Overview

This project uses a combination of:
- Natural Language Processing (NLP)
- Feature engineering (TF-IDF, OneHot Encoding)
- Supervised Machine Learning models
- FastAPI backend for serving predictions
- Database integration for storing notification logs

----------------------------------------------------------------------------------------------------------------------------------------
Features

- Notification classification using ML models
- TF-IDF based text feature extraction
- Categorical encoding using OneHotEncoder
- FastAPI-based prediction API
- Model retraining pipeline support
- MongoDB integration for data storage
- Priority scoring system for notifications

----------------------------------------------------------------------------------------------------------------------------------------
Tech Stack

- Python
- Scikit-learn
- Pandas & NumPy
- TF-IDF Vectorizer 
- OneHotEncoder
- FastAPI
- MongoDB

---------------------------------------------------------------------------------------------------------------------------------------
Project Structure


Smart-Notification-Prioritizer---NotifAI
│
├── api_integration.py                      # FastAPI backend endpoints
├── frontend.py                             # Frontend interface (if applicable)
├── processing.py                           # Preprocessing & feature engineering
├── test_fun.py                             # Utility/test functions
├── db_config.py                            # MongoDB configuration
│
├── model1.pkl                              # Primary ML model
├── model2.pkl                              # Secondary ML model
├── test_retrain_model.pkl                  # Retrainable model for pipeline check purposes
│
├── tfidf_vectorizer.pkl                    # Text vectorizer
├── onehot_encoder_app.pkl                  # App feature encoder
├── onehot_encoder_category.pkl             # Category encoder
├── onehot_encoder_time.pkl                 # Time-based encoder
|__ onehot_encoder_app.pkl                  # App feature encoder for model 2
│
├── feature_order_model1.pkl                # Feature order tracker model 1
├── feature_order_model2.pkl                # Feature order tracker model 2
├── feature_order_model2_for_retrain.pkl    # Feature order tracker retrain model
│
├── model1_priority_dataset.csv             # Training dataset (synthetic dataset)
├── model2_priority_dataset.csv
├── notifai_final_realistic_dataset.csv
│
└── data_analysis.ipynb                     # EDA & experimentation notebook

----------------------------------------------------------------------------------------------------------------------------------------
 



