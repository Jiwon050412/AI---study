# BERT Fine-tuning for IMDb Sentiment Classification

This project fine-tunes a pretrained BERT model to classify movie reviews from the IMDb dataset as positive or negative.

# Goal

The goal of this project is to understand the overall process of fine-tuning a pretrained Transformer model for a supervised NLP classification task.

# Dataset
- Dataset: IMDb Movie Reviews
- Task: Binary Sentiment Classification
- Label 0: Negative
- Label 1: Positive

The original IMDb training dataset was divided into training and validation sets, while the original test dataset was kept separate for final evaluation.

# Model
- Model: bert-base-uncased
- Architecture: BERT Encoder + Classification Head
- Number of classes: 2

The pretrained BERT model provides general language representations, while the classification head is trained to predict the sentiment of each review.

# Data Preparation

The dataset was processed through the following steps:

1. Load the IMDb dataset
2. Shuffle and select a subset of the training and test data
3. Split the training data into training and validation sets
4. Tokenize the reviews using the BERT tokenizer
5. Pad and truncate sequences to a maximum length of 128
6. Convert the dataset into PyTorch tensors
7. Rename the label column to labels for compatibility with the Hugging Face model

# Fine-tuning

The model was trained using a manual PyTorch training loop rather than the Hugging Face Trainer API.

The training process includes:

- Forward pass
- Loss calculation
- Backpropagation
- Parameter update using AdamW

Hyperparameters:

- Batch size: 16
- Learning rate: 2e-5
- Epochs: 3
- Maximum sequence length: 128
- Validation Results
| Epoch |	Train Loss	| Validation Loss	| Validation | Accuracy |
| 1	| 0.4221	| 0.3538	| 85.0% |
| 2	| 0.2285	| 0.3308	| 86.5% |
| 3	| 0.1132	| 0.4396	| 86.3% |

The validation results show that the model achieved its best performance at Epoch 2.

Although the training loss continued to decrease during Epoch 3, the validation loss increased and validation accuracy decreased. This indicates that the model began to overfit the training data.

# Final Test Results

The model with the best validation performance was saved and evaluated on the untouched IMDb test dataset.

- Test Loss: 0.3593
- Test Accuracy: 85.5%

The test dataset was not used during training or model selection and was reserved for the final evaluation.

# Evaluation

The model was evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score

The confusion matrix was used to analyze the types of classification errors made by the model, while precision, recall, and F1-score provided additional evaluation metrics.

# Key Concepts Learned

Through this project, I learned:

- Supervised learning
- Train / validation / test split
- Tokenization
- BERT input representation
- Fine-tuning pretrained models
- Classification heads
- Manual PyTorch training loops
- Validation and test evaluation
- Overfitting
- Model checkpointing
- Confusion matrices
- Precision, Recall, and F1-score
- GPU training with CUDA
- Hugging Face Transformers

# Project Structure
bert-finetuning/
├── README.md
└── bert_finetuning.ipynb

# Tools
- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- scikit-learn
- Kaggle
