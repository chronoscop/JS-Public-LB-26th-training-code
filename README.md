# JS training code

## Introduction

This repository contains the training code for the **Jane Street Real-Time Market Data Forecasting** competition.

## Installation Requirements

Ensure you have the following dependencies installed:

- Python 3.8 or 3.11
- Required Python libraries (can be installed via `requirements.txt`)

**Hardware Requirements**:

- **RAM**: At least 60GB to 100GB of RAM is recommended for optimal performance during training, especially when dealing with large datasets.
- **GPU**: A GPU with at least 16GB of VRAM is recommended for training
```
pip install -r requirements.txt
```

## Dataset

You can download the dataset for the competition from [this link](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data).

The dataset should be placed in the `./data` directory, or adjust the paths in the code accordingly.

In the training code, we utilize lag features to enhance the model's performance. To generate these lag features, we provide a Python script named `janestree_process_data.py`, which processes the dataset and adds the necessary lag features. Make sure to run this script before starting the training process or you can visit  https://www.kaggle.com/code/motono0223/js24-preprocessing-create-lags for detail.

## Usage 

**Note**: Since we are open-sourcing the code, we have not modified the `--data_path` and `--output_dir` parameters in the training script. You will need to manually adjust these paths in the command to point to your dataset and desired model save location.

### Our Approach

We shared our approach detail in a Kaggle discussion post. You can find the detailed explanation and the link to the full code here: [[Public LB 26th] TabM, AutoencoderMLP with online training & GBDT offline models](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556610).

## Code Structure

- `jane_street_tabm_training.py`: The script to train the tabM.
- `js_xgb_training.py`: The script to train the xgb.
- `nn_mse.py`: The script to train the AutoencoderMLP.
- `dataset/janestree_process_data.py`: The script to create in dataset.

## License

This project is licensed under the [MIT License](https://github.com/chronoscop/JS-Public-LB-26th-training-code/blob/main/LICENSE).

## Contributing

Feel free to open issues or submit pull requests! If you have any questions or suggestions, you can reach out via GitHub issues or the [Kaggle discussion](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556610).
