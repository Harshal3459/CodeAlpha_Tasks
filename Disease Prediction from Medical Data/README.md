# Heart Disease Prediction using Logistic Regression

This project implements a heart disease prediction model using Logistic Regression. The dataset is obtained from Kaggle and processed using Python with scikit-learn.

## Dataset
The dataset used in this project is available on Kaggle:
[Heart Diseases Prediction Dataset](https://www.kaggle.com/code/figolm10/heart-diaseases-prediction/input)

## Installation
To run this project, ensure you have the following dependencies installed:

```bash
pip install numpy pandas scikit-learn google-colab
```

## Project Structure
- `heart_disease_prediction.ipynb`: Jupyter Notebook containing the code.
- `heart.csv`: The dataset file.
- `README.md`: Project documentation.

## Features Used
The dataset contains various medical attributes including:
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol Level
- Fasting Blood Sugar
- Resting ECG Results
- Maximum Heart Rate Achieved
- Exercise-Induced Angina
- ST Depression Induced by Exercise
- Slope of the Peak Exercise ST Segment
- Number of Major Vessels Colored by Fluoroscopy
- Thalassemia Type

## Model Training
The dataset is split into training (80%) and testing (20%) sets. A Logistic Regression model is trained to classify whether a patient has heart disease or not.

## Usage
To run the model:
1. Load the dataset from Google Drive.
2. Preprocess and split the data.
3. Train the Logistic Regression model.
4. Make predictions based on input features.

### Example Prediction:
A sample input to the model:
```python
input_data = np.array([[62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2]])
prediction = model.predict(input_data)
```
If the output is `0`, the person does **not** have heart disease; if `1`, the person **has** heart disease.

## Accuracy
- Training Accuracy: ~85% (varies based on training runs)
- Testing Accuracy: ~82% (varies based on dataset and model parameters)

## Contributing
Feel free to fork this repository and contribute by adding improvements such as:
- Hyperparameter tuning
- Additional ML models
- Web app interface

## License
This project is for educational purposes and does not provide medical advice. Always consult a healthcare professional for diagnosis and treatment.
