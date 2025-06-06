# Keystroke Prediction from EMG Signals

This project explores keystroke prediction using surface electromyography (sEMG) signals to enable keyboardless human-computer interaction. The primary goal is to minimize **Character Error Rate (CER)** through architectural design, preprocessing, and augmentation strategies.

> This project is built upon the [emg2qwerty](https://github.com/facebookresearch/emg2qwerty) work from Meta. For more details, refer to their [research paper](https://arxiv.org/abs/2410.20081).

## 📌 Project Overview

This project is built upon the emg2qwerty work from Meta, implementing various deep learning architectures for predicting keystrokes from surface electromyography (sEMG) signals. The goal is to develop an accurate model that can predict which key was pressed based on the muscle activity patterns captured by sEMG sensors.

### Key Objectives
- Decode keystrokes from wrist-worn EMG sensors
- Minimize Character Error Rate (CER)
- Evaluate performance for single-user, multi-user, and unseen-user scenarios
- Implement and compare various neural network architectures
- Optimize data preprocessing and augmentation techniques

## 🧪 Methodology

### 1. Data Preprocessing
- Input data consists of sEMG signals from 2 bands with 16 electrode channels each
- Spectrogram normalization is applied per electrode channel per band
- Data is windowed and padded for training/validation
- Full session data is used for testing without windowing
- Native 2 KHz sampling with 128-bin STFT emphasizing 0-300 Hz range

### 2. Model Architectures

1. **Best Architecture (GRU + CNN)**
   - Located in: `Project-CNN-GRU-LL-RELU-Norm-Droput-p-0.2-/`
   - Combines CNN and GRU layers for optimal performance
   - Uses ReLU activation
   - Includes normalization and dropout (p=0.2)
   - Key components:
     - Time-Depth Separable (TDS) convolutional encoder
     - GRU layers for sequence modeling
     - Layer normalization
     - Dropout for regularization
   - Achieves best test CER of 14.415%

2. **GRU Architecture**
   - Located in: `Project-Best_Architecture_GRU_150_Epoch/`
   - Uses GRU (Gated Recurrent Unit) architecture
   - Trained for 150 epochs
   - Key components:
     - Spectrogram normalization
     - Multi-band rotation invariant MLP
     - GRU layers for sequence processing
     - Dropout for regularization

3. **Bidirectional RNN**
   - Located in: `Project-BidirectionalRNN_HiddenSize_128_NumLayers_4/`
   - Uses bidirectional RNN architecture
   - Hidden size: 128
   - Number of layers: 4
   - Processes sequences in both forward and backward directions

4. **LSTM Architecture**
   - Located in: `Project-BEST_ARCH_1_30_EPOCH_LSTM/`
   - Uses LSTM (Long Short-Term Memory) architecture
   - Trained for 30 epochs
   - Good for capturing long-term dependencies

### 3. Testing Scenarios

1. **Single-User Testing**
   - Benchmarked various models: CNNs, RNNs (LSTM, GRU), CNN-RNN hybrids, and Transformers
   - Tested preprocessing methods (resampling, STFT, spectrogram bins)
   - Explored data augmentation (band rotation, temporal jitter, spectral masking)

2. **Multi-User Testing**
   - Trained best-performing model on 3 users
   - CER increased modestly due to variance across users:
     - **Test CER**: 19.718

3. **Unseen-User Testing**
   - Trained on two users, tested on a third unseen user
   - Significant CER increase observed:
     - **Test CER**: 71.055
   - Indicates poor generalization without additional regularization

## 🏆 Results

### Architecture Performance Comparison

| Architecture                                     | Validation CER | Test CER  |
|--------------------------------------------------|----------------|-----------|
| **GRU + CNN**                                    | **14.532**     | **14.415** |
| GRU + CNN + Linear + ReLU + Dropout + Norm       | 13.469         | 15.582    |
| GRU + CNN + Linear + ReLU + Norm (no dropout)    | 13.491         | 15.841    |
| GRU + CNN (no augmentation)                      | 12.073         | 16.771    |
| **Vanilla Baseline CNN**                         | 19.650         | 22.670    |

> 🚀 **Best overall test CER:** 14.415  
> 🧪 **Improvement over baseline:** ~36% reduction in test CER

## 📊 Key Takeaways
- **Best Model**: GRU + CNN (with dropout, normalization, and ReLU)
- **Best Preprocessing**: Native 2 KHz sampling, 128-bin STFT emphasizing 0-300 Hz range
- **Augmentation**: Improves generalization but reduces single-user accuracy
- **Transformers**: Underperformed without optimization
- **Generalization**: Requires broader data and better augmentation strategies

## 🛠️ Implementation Details

### Key Files
1. **modules.py**
   - Core model architecture definitions
   - Custom layers and components
   - Model building blocks

2. **lightning.py**
   - PyTorch Lightning implementation
   - Training and validation loops
   - Data loading and preprocessing

3. **config/*.yaml**
   - Model hyperparameters
   - Training configuration
   - Data split settings

### Training Features
- PyTorch Lightning framework
- Custom data loaders with windowing
- Character Error Rate metrics
- Learning rate scheduling
- Gradient clipping
- Early stopping

## 🧠 Conclusion
GRU-CNN hybrids yield strong results in personalized settings but face challenges generalizing to unseen users. Future work should prioritize larger, more diverse datasets and robust regularization to enhance deployment viability for assistive interfaces.

## 📋 Requirements

- Python 3.8+
- PyTorch
- PyTorch Lightning
- NumPy
- Pandas
- scikit-learn

## 🚀 Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd sEMG_Keystroke_prediction
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Code

Each architecture implementation can be run independently. Here's how to run the best performing model (GRU + CNN):

1. Navigate to the best architecture directory:
```bash
cd Project-CNN-GRU-LL-RELU-Norm-Droput-p-0.2-
```

2. Run the training script:
```bash
python train.py
```

For other architectures, follow the same pattern by navigating to their respective directories and running the training script.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.