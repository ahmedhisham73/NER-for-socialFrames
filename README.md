# Documentation: Challenges in NER for Social Bias Frames
# Introduction
The task at hand was to conduct Named Entity Recognition (NER) on the Social Bias Frames dataset. NER is a pivotal preliminary task for many NLP applications, aiming to classify named entities in text into predefined categories.

# 1.  Exploratory Data Analysis (EDA)
Null Values: The initial steps of EDA revealed the presence of null values within the dataset. Handling these was essential to ensure that the subsequent stages of modeling were not impacted adversely.

Class Distribution: A glaring class imbalance was observed when examining the distribution of the 'targetCategory'. Such imbalances can lead a model to show biased results towards the majority class.

# 2. Unique Categories and Class Imbalances
Unique Categories: The nature of the 'targetCategory' was such that some entries combined multiple categories. This increased the complexity of the classification task.

Handling Class Imbalance: The initial approach to address the imbalance involved downsampling the majority classes. This was supplemented by using augmentation techniques for minority classes. Techniques deployed included random word swaps and using synonyms.

# 3. Outliers and Data Cleaning
Category Representation: Entries in 'targetCategory' were found to be represented as lists, such as ['culture', 'victim']. To streamline the process, these were converted to a more standard string representation.

Outlier Management: Given the vast array of unique values in 'targetCategory', a strategic decision was made to focus on the primary categories. Remaining ones were treated as outliers to sharpen the focus and improve model performance.

# 4. Baseline Modeling - Naïve Bayes
Setting the Baseline: A Naïve Bayes classifier was employed to set a baseline for the task. It yielded the following scores:
Precision: 0.77 (Gender), 0.73 (Race), 0.79 (Culture), 0.81 (Victim)
Recall: 0.80 (Gender), 0.79 (Race), 0.84 (Culture), 0.65 (Victim)


# 5. Advanced Modeling - DistilBERT
Migration to Transformers: Traditional models like Naïve Bayes showed limitations in capturing the nuances of the dataset. This led to the adoption of DistilBERT, a lighter variant of the original BERT model.

Performance of DistilBERT:

Final Validation Precision: 0.7585
Final Validation Recall: 0.9166
Challenges with DistilBERT:

Class Weights: To address the class imbalance in the dataset, class weights were introduced. However, integrating these with the model posed challenges initially.
Model Metrics: Alongside accuracy, other metrics like precision, recall, and F1 scores were considered to offer a more holistic view of the model's performance.
Hyperparameter Tuning: Determining the best parameters like learning rate, batch size, and epochs required multiple iterations.
Callback Implementations: To further enhance the model's training phase, callbacks like early stopping and model checkpointing were introduced.


# Conclusion
The journey of conducting NER on the Social Bias Frames dataset was interspersed with numerous challenges, right from the initial data handling to advanced modeling. By leveraging state-of-the-art models like DistilBERT and adopting a strategic approach, these challenges were successfully navigated. As a future direction, delving into more powerful transformer models, ensemble techniques, and further hyperparameter optimization can be considered.
