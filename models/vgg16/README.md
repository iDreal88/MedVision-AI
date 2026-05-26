# Mammogram Analysis Project

This project adapts a breast cancer detection model (VGG16 + CLAHE) for local execution.

## Setup

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Troubleshooting

### AVX Error (Crash on start)
If you see an error like `The TensorFlow library was compiled to use AVX instructions, but these aren't available on your machine`, the script has been updated to use `tensorflow-cpu` which should resolve this on older Intel Macs.

2.  **Prepare Dataset**
    Place your dataset in a folder named `dataset` in this directory.
    The expected structure is:
    ```
    dataset/
    ├── Augmented Dataset/
    │   ├── Cancer/
    │   └── Non-Cancer/
    └── Original Dataset/
        ├── Cancer/
        └── Non-Cancer/
    ```
    *Note: If you have a flat structure (just `Cancer` and `Non-Cancer`), place them directly inside `dataset/` (e.g. `dataset/Cancer`) and the script will try to detect them.*

3.  **Run the script**
    To use your Apple M2 GPU, run the script with the native Python 3.9 interpreter:
    ```bash
    /usr/bin/python3 main.py
    ```

4.  **Outputs**
    Check the `output/` folder for generated plots and result images:
    - `01_samples.png`: Sample processed images.
    - `02_confusion_matrix.png`: Model performance matrix.
