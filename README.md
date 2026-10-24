# PulseSense

## Heart health screening app

PulseSense is a simple Streamlit app that uses a trained machine-learning model to provide a heart-disease risk screening signal from health measurements.

The app is designed to be easy to use:

1. Enter the requested health information.
2. Select **Run my heart health check**.
3. Review the model's screening signal.

> **Important medical notice**  
> PulseSense is for education and early screening only. It is not a diagnosis and must not replace a doctor, medical test, or emergency care. If you have severe or urgent symptoms, contact emergency services immediately.

## Start here

### 1. Install Python

Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/).

### 2. Open the project folder

Open PowerShell or a terminal in the folder containing `app.py`.

### 3. Create a virtual environment

Using a virtual environment keeps this project separate from other Python projects.

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the environment directly without activating it:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### 4. Install the dependencies

```powershell
python -m pip install -r requirements.txt
```

The dependency list is stored in [requirements.txt](./requirements.txt).

### 5. Launch PulseSense

```powershell
streamlit run app.py
```

Open the local address shown in the terminal. It is usually:

```text
http://localhost:8501
```

To stop the app, return to the terminal and press `Ctrl+C`.

## What the app asks for

PulseSense collects the following information:

| Section | Information |
| --- | --- |
| Personal details | Age and sex |
| Basic measurements | Resting blood pressure and cholesterol |
| Heart signals | Chest pain type, resting ECG, maximum heart rate, and exercise-induced angina |
| Additional indicators | Fasting blood sugar, ST depression, and ST-segment slope |

Use the values from a trusted medical measurement or report whenever possible. Do not guess values if you are using the result to discuss your health with a clinician.

## Understanding the result

After submission, the app displays:

- A summary of selected values
- A model signal percentage when available
- Either an **Elevated risk signal** or **Lower risk signal** message

The result describes what the model sees in the supplied data. It does not confirm whether a person has or does not have heart disease.

## How it works

The prediction pipeline is:

```text
User input
    -> categorical values converted to model columns
    -> missing columns filled with zero
    -> columns reordered to training order
    -> numerical values standardized
    -> KNN model prediction
    -> screening result shown in the app
```

The app uses a saved K-Nearest Neighbors classifier. The model compares a new input with similar examples from its training data and predicts the most common class among its nearest neighbors.

## Files in this project

| File | Purpose |
| --- | --- |
| [app.py](./app.py) | Streamlit interface and prediction pipeline |
| [KNN_heart.pkl](./KNN_heart.pkl) | Trained KNN model |
| [scaler.pkl](./scaler.pkl) | Fitted feature scaler |
| [columns.pkl](./columns.pkl) | Expected model feature order |
| [requirements.txt](./requirements.txt) | Python dependencies |
| [README.md](./README.md) | Project guide |

## Troubleshooting

### `streamlit` is not recognized

Run Streamlit through the virtual environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### A package is missing

Reinstall the dependencies:

```powershell
python -m pip install -r requirements.txt
```

### The browser page does not open

Copy the URL printed in the terminal and open it manually. The default address is `http://localhost:8501`.

### The model raises a loading or compatibility warning

The model files were serialized separately from the application. Loading them with the same scikit-learn version used during training is recommended. If the original training environment is unavailable, retrain and save the model again with the current environment.

## Technical details

The saved model is a `KNeighborsClassifier` with:

- `n_neighbors = 5`
- Uniform neighbor weighting
- Minkowski distance
- Euclidean distance configuration with `p = 2`
- 15 processed input features

The application uses `StandardScaler` before prediction and caches the model artifacts for faster Streamlit reruns.

## Project limitations

- The original raw dataset is not included.
- The training script is not included.
- Accuracy, precision, recall, F1-score, ROC-AUC, and confusion-matrix results are not currently documented.
- The model may not generalize to every population or healthcare setting.
- A model probability is not the same as a medical probability or diagnosis.

For a stronger academic or production project, add the dataset source, data dictionary, training notebook, evaluation report, dependency versions, and model validation results.

## License

This project does not currently include a license file. Add a license before distributing or publishing it.
