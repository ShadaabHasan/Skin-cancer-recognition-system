# Skin Cancer Recognition System

A computer vision project that classifies images of skin lesions into **7 types**, including melanoma, using convolutional neural networks (CNNs). It includes a simple desktop app where you upload an image and get a prediction.

📄 **Published:** "Skin Cancer Recognition System", *Journal of Emerging Technologies and Innovative Research (JETIR)*, 29 June 2024. [Read the paper](https://www.jetir.org/view?paper=JETIR2406947)

## Dataset

[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000): **10,015** dermatoscopic images of skin lesions. This project uses the small version of the images (28 × 28 pixels, RGB), stored as `hmnist_28_28_RGB.csv`.

The 7 classes:

| Label | Code | Lesion type |
|---|---|---|
| 0 | akiec | Actinic keratoses and intraepithelial carcinoma |
| 1 | bcc | Basal cell carcinoma |
| 2 | bkl | Benign keratosis-like lesions |
| 3 | df | Dermatofibroma |
| 4 | nv | Melanocytic nevi |
| 5 | vasc | Vascular lesions |
| 6 | mel | Melanoma |

## Method

1. **Balance the classes:** the dataset is very unbalanced (most images are `nv`), so the rare classes were oversampled with `RandomOverSampler`, giving 46,935 images.
2. **Split:** 80% training, 20% testing.
3. **Compare 4 CNN architectures**, trained with the Adam optimiser, categorical cross-entropy loss, learning-rate reduction and early stopping.
4. **Choose the best:** Model 1 performed best, so it was retrained and saved as `finalModel.h5`.
   - Model 1: four convolutional layers (32, 32, 64, 64 filters), max pooling, dropout (0.25), a dense layer of 128 units, and a 7-class softmax output.
5. **Desktop app:** `GUI.py` (Tkinter) lets you upload an image, resizes it to 28 × 28, and shows the predicted lesion type.

## Tech stack

- Python
- TensorFlow / Keras
- scikit-learn, imbalanced-learn
- NumPy, Pandas, Matplotlib, Seaborn
- Tkinter and Pillow (desktop app)

## How to run

**Notebook:**
1. Download `hmnist_28_28_RGB.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and put it in the project folder. (The file in this repo is an empty placeholder.)
2. Open `final_notebook.ipynb` and run all cells.

**Desktop app:**
```bash
pip install tensorflow pillow numpy
python3 GUI.py
```
Click **Upload**, choose an image, then click **Start**.

## Limitations

- **The reported test accuracy is too optimistic.** The data was oversampled *before* it was split, so copies of the same image can appear in both the training and the test set. The test set was also used for early stopping. Accuracy on truly new images would be lower.
- The images are only 28 × 28 pixels, which loses a lot of detail that matters for diagnosis.
- Accuracy alone is not a good measure here. For skin cancer, **melanoma recall** (how many real melanomas the model catches) matters most.
- This is a student project and is **not a medical tool**.

## Planned improvements

- Split the data first, then oversample only the training set, and keep a separate validation set.
- Use transfer learning (for example EfficientNet) on full-size images.
- Report per-class recall, especially for melanoma, and a confusion matrix.
- Add Grad-CAM heatmaps to show which parts of the image the model looks at.

## Team

Mini project, TE Computer Engineering (Div A), Rizvi College of Engineering, 2023–2024.

- **Shadaab Hasan** (Team Leader)
- Siddharth Pallar
- Ayan Mastim
- Hamza Khan

**Guide:** Prof. Mohd Ashfaque Shaikh
