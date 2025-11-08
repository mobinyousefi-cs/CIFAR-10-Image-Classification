# CIFAR-10 Image Classification with Keras

Author: **Mobin Yousefi** · [GitHub: mobinyousefi-cs](https://github.com/mobinyousefi-cs)

---

## 📌 Project Overview

This project implements an end-to-end **image classification pipeline** on the **CIFAR-10** dataset using a **Convolutional Neural Network (CNN)** built with **TensorFlow / Keras**.

The goal is to provide a **clean, modular, and production-ready** codebase that you can:

- Use as a **template** for future image classification projects.
- Extend with **deeper architectures, regularization techniques, and experiments**.
- Integrate into a larger **MLOps / research workflow**.

The CIFAR-10 dataset contains **60,000 RGB images** of size **32×32×3** across **10 classes**:

> airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Keras provides CIFAR-10 via `tf.keras.datasets.cifar10`, so there is **no manual download required**.

---

## 🧱 Repository Structure

```text
cifar10-image-classification/
├── src/
│   └── cifar10_image_classification/
│       ├── __init__.py
│       ├── config.py            # Central configuration & constants
│       ├── data.py              # Data loading & tf.data pipelines
│       ├── model.py             # CNN model architecture & compilation
│       ├── train.py             # Training loop, callbacks, checkpoints
│       ├── evaluate.py          # Evaluation on the test set
│       ├── predict.py           # Single-image prediction CLI
│       └── utils/
│           ├── seed.py          # Reproducibility utilities
│           └── logger.py        # Logging configuration
│
├── tests/
│   ├── test_data.py             # Basic tests for data pipeline
│   └── test_model.py            # Basic tests for model construction
│
├── artifacts/                   # (Created at runtime) models, logs, etc.
│   ├── models/
│   └── logs/
│
├── README.md
├── pyproject.toml               # Project metadata & dependencies
├── LICENSE                      # MIT License
└── .gitignore
```

---

## 📚 CIFAR-10 Dataset

- **Dataset homepage:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Train images:** 50,000
- **Test images:** 10,000
- **Image size:** 32×32×3 (RGB)
- **Classes (10):**
  - airplane
  - automobile
  - bird
  - cat
  - deer
  - dog
  - frog
  - horse
  - ship
  - truck

In this project:

- We load CIFAR-10 using `tf.keras.datasets.cifar10`.
- We **normalize** pixel values to `[0, 1]`.
- We split part of the training data into a **validation set**.
- We build efficient `tf.data.Dataset` pipelines for training, validation, and testing.

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/mobinyousefi-cs/cifar10-image-classification.git
cd cifar10-image-classification
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Install the project in editable mode

This will install all runtime dependencies defined in `pyproject.toml`:

```bash
pip install --upgrade pip
pip install -e .
```

> **Note:** You need a working TensorFlow installation with GPU support if you want faster training. Otherwise, CPU-only is also fine for experimentation.

---

## 🧠 Model Architecture

The default model is a **moderately deep CNN** suitable for CIFAR-10:

- Input: 32×32×3 images
- Data preprocessing with `Rescaling(1./255)`
- Optional data augmentation (RandomFlip, RandomRotation)
- Several convolutional blocks:
  - `Conv2D → BatchNorm → ReLU → MaxPooling → Dropout`
- Final dense layers with `Softmax` output over 10 classes

The model is compiled with:

- **Loss:** `sparse_categorical_crossentropy`
- **Optimizer:** `Adam`
- **Metrics:** `accuracy`

Hyperparameters (batch size, learning rate, epochs, etc.) are defined in `config.TrainingConfig` and can be overridden via CLI arguments.

---

## ▶️ How to Train

From the project root:

```bash
python -m cifar10_image_classification.train
```

Optional arguments:

```bash
python -m cifar10_image_classification.train \
  --epochs 40 \
  --batch-size 128 \
  --learning-rate 0.0005 \
  --model-dir artifacts/models/exp1
```

During training, the script will:

- Load CIFAR-10 and build `tf.data` pipelines.
- Construct the CNN model.
- Train with **EarlyStopping** and **ModelCheckpoint**.
- Save the best model to `artifacts/models/best_model.h5` (by default).
- Log metrics to `artifacts/logs/` (TensorBoard-compatible logs).

You can inspect training curves with TensorBoard:

```bash
tensorboard --logdir artifacts/logs
```

---

## 📊 How to Evaluate

To evaluate a saved model on the CIFAR-10 **test set**:

```bash
python -m cifar10_image_classification.evaluate \
  --model-path artifacts/models/best_model.h5
```

This will print metrics such as:

- Test loss
- Test accuracy

and can be extended to include:

- Confusion matrix
- Per-class accuracy
- Classification report

---

## 🔍 How to Predict on a Single Image

You can run inference on a **single image file**:

```bash
python -m cifar10_image_classification.predict \
  --model-path artifacts/models/best_model.h5 \
  --image-path path/to/image.png
```

The script will:

1. Load the image from disk.
2. Resize it to 32×32.
3. Normalize it to `[0, 1]`.
4. Run the model.
5. Print the **predicted class name** and probabilities.

> Note: CIFAR-10 images are low-resolution. If you use higher-resolution images, they will be downscaled to 32×32, which may affect prediction quality.

---

## 🧪 Running Tests

This project includes a few basic **pytest** tests to validate data and model wiring:

```bash
pip install pytest
pytest
```

- `tests/test_data.py` – sanity checks for dataset shapes and splits.
- `tests/test_model.py` – ensures the model builds and performs a forward pass on dummy data.

These tests are intentionally lightweight and can be extended as you evolve the project.

---

## 🧩 Extending the Project

Some ideas for further experimentation:

- Add **deeper CNN architectures** (e.g., ResNet-like blocks).
- Use **learning rate schedules** or **cosine annealing**.
- Add **label smoothing**, **Mixup**, or **Cutout** regularization.
- Apply **data augmentation strategies** specific to CIFAR-10.
- Implement **experiment tracking** (Weights & Biases, MLflow, etc.).
- Port the core logic into a **Jupyter Notebook** for educational demos.

Because the codebase is modular (`config.py`, `data.py`, `model.py`, `train.py`), you can safely experiment with any component without breaking the rest.

---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 🙌 Acknowledgements

- **CIFAR-10 Dataset** – Collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton at the University of Toronto.
- **TensorFlow / Keras** – For providing high-level deep learning APIs.

If you use or extend this repository, a ⭐ on GitHub is always appreciated!

