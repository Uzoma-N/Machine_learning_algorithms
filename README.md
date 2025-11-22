# Machine_learning_algorithms

Project collection of small Python examples and notebooks implementing common machine learning tasks and demos. Each script focuses on a single problem and demonstrates typical data preprocessing, model design, training and simple evaluation.

**Repository layout**
- `cnn_for_face_mask_detection.py`: Example using convolutional neural networks (CNNs) to detect face masks in images. Typical techniques: image loading and augmentation, CNN model definition with Keras, training loop, validation, and saving model weights.
- `named_entity_recognition.py`: Demonstration of Named Entity Recognition (NER) on a small dataset. Techniques used: tokenization, sequence labeling (e.g., BiLSTM+CRF or transformer-based fine-tuning), dataset preprocessing, label encoding, and evaluation with precision/recall/F1.
- `profit_prediction.py`: A tabular regression example predicting startup profit from features in `datasets/startups.csv`. Techniques used: data cleaning, exploratory data analysis (basic), feature engineering, model selection (linear regression / tree-based), cross-validation, and metric reporting (MSE / RMSE).
- `time_series_with_LSTM.py`: Time-series forecasting using LSTM networks on `datasets/airline_passengers.csv`. Techniques used: sliding-window sequence creation, scaling, LSTM model definition with Keras, training, forecasting and plotting predictions vs. ground truth.

**Datasets**
- `datasets/airline_passengers.csv` — classic monthly passenger counts time series (used by `time_series_with_LSTM.py`).
- `datasets/ner_dataset.csv` — small NER dataset used by `named_entity_recognition.py`.
- `datasets/startups.csv` — tabular dataset for `profit_prediction.py`.

**Environment & dependencies**
- The repository contains a virtual environment at `machine_learning_algorithm/` (do not commit or modify it unless you intend to). For reproducible setups, create a fresh virtual environment and install dependencies.
- Recommended Python: use a modern 3.x interpreter. If you plan to use TensorFlow (for the CNN and LSTM scripts), check the exact TensorFlow release compatibility with your Python version — for example, TensorFlow 2.20 has specific supported Python versions; consult the official TensorFlow release notes or PyPI page for exact supported versions.

Quick setup (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
# optional: create requirements.txt from your environment and then install
# pip install -r requirements.txt
```

If you want a minimal set of typical packages for these examples, you can install:
```powershell
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow torch transformers
```
Adjust packages as needed (e.g., `tensorflow` vs `tensorflow-cpu`, `torch` versions) depending on your hardware and Python version.

**How to run each script**
- `cnn_for_face_mask_detection.py`: Prepare an images directory as expected by the script (or adapt the paths). Run `python cnn_for_face_mask_detection.py`. Monitor training output and model saving.
- `named_entity_recognition.py`: Ensure `datasets/ner_dataset.csv` is present. Run `python named_entity_recognition.py` to train/evaluate the sequence labeling model.
- `profit_prediction.py`: Run `python profit_prediction.py`. The script reads `datasets/startups.csv`, trains a regression model, and prints evaluation metrics.
- `time_series_with_LSTM.py`: Run `python time_series_with_LSTM.py`. The script will train an LSTM on `datasets/airline_passengers.csv` and (optionally) produce plots of forecasts.

**Notes & next steps**
- Add a `requirements.txt` by running `pip freeze > requirements.txt` in a controlled virtual environment to capture exact versions.
- For GPU-enabled TensorFlow or PyTorch, follow official installation instructions for your OS and CUDA/cuDNN versions.
- To make examples more robust, consider adding CLI flags (argparse) to set dataset paths, model hyperparameters, and whether to train or only evaluate.

---
Repository generated README — contact the author or open an issue for corrections.
