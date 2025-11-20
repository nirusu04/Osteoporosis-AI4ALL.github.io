# AI-Based Osteoporosis Detection Using X-Ray Imaging

Built an AI-driven computer vision pipeline to detect early osteoporosis using X-ray images. Applied data cleaning, transfer learning, and CNN/ResNet architectures to train a robust model capable of achieving 85%+ accuracy, developed as part of the AI4ALL Ignite program.


## Problem Statement <!--- do not change this line -->

Osteoporosis is a widespread but often undiagnosed condition, affecting over 200 million people globally. Early detection is difficult because the disease is mostly “silent” until fractures occur. This project aims to determine whether machine learning and medical imaging can reliably predict osteoporosis risk in adults aged 35–80 using bone density measurements, demographic variables, and X-ray imaging.
Achieving accurate early prediction has the potential to improve clinical decision-making, reduce fracture risk, and make screening more accessible.

## Key Results <!--- do not change this line -->

 -Performed extensive data cleaning and preprocessing on real-world osteoporosis X-ray datasets.

 -Implemented CNNs and transfer learning with ResNet, training models to classify osteoporosis-related features.

 -Achieved ≥85% accuracy on preliminary classification tasks using image-based bone density patterns.

 -Identified critical sources of bias such as imbalanced age groups, inconsistent imaging quality, and underrepresentation across demographics.

 -Applied techniques such as SMOTE oversampling, quality filtering, and stratified sampling to mitigate dataset bias.



## Methodologies <!--- do not change this line -->


To build a reliable prediction model, we:

Cleaned and preprocessed multiple osteoporosis X-ray datasets, removing poor-quality and inconsistent images.

Applied transfer learning using pretrained ResNet architectures to enhance performance on limited datasets.

Trained and evaluated CNN-based classifiers, tuning hyperparameters to optimize accuracy and reduce overfitting.

Experimented with Random Forest, Logistic Regression, and XGBoost for structured clinical data (if available).

Compared model performance across algorithms to ensure reliability and generalizability.

Identified and addressed dataset bias through augmentation, SMOTE, and demographic balancing techniques.


## Data Sources <!--- do not change this line -->

Multi-Class Knee Osteoporosis X-Ray Dataset https://www.kaggle.com/datasets/mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset/data

## Technologies Used <!--- do not change this line -->

-Python
-NumPy & pandas
-TensorFlow / Keras
-PyTorch (optional depending on your implementation)
-CNNs
-Transfer Learning (ResNet)
-Scikit-learn
-SMOTE (imbalanced-learn library)
-Matplotlib / Seaborn
-Google Colab


## Authors <!--- do not change this line -->


*EXAMPLE:*
*This project was completed in collaboration with:*
- Sampada Niroula
- Dia Poudel
- Michael Montanez
- Sydney Eze
