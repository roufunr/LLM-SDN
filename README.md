# Packet Info Predictor with Transformer Model

This project implements a machine learning system that predicts packet information for the next k packets based on previous packet history. It uses a transformer model (GPT-2) with reinforcement learning for continuous improvement.

## Problem Statement

Given a sequence of network packets with their "Info" field, predict the next k packets' information. The system learns from its predictions and improves over time using reinforcement learning principles.

## Architecture Overview

### 1. Model Choice: Transformer (GPT-2)
**Why Transformer?**
- **Sequential Dependencies**: Packet info has temporal relationships
- **Variable Length**: Info strings vary in length and complexity
- **Context Understanding**: Transformers excel at understanding context from previous sequences
- **Text Generation**: Perfect for predicting next packet info sequences
- **Pre-trained Knowledge**: GPT-2 comes with pre-trained language understanding

### 2. Data Flow Architecture

```
Input: [Packet 1, Packet 2, ..., Packet 1000] (Context Window)
       ↓
Transformer Model (GPT-2)
       ↓
Output: [Predicted Packet 1001, ..., Packet 1010] (Next 10 packets)
       ↓
Compare with Actual: [Actual Packet 1001, ..., Packet 1010]
       ↓
Calculate Bit-by-Bit Accuracy
       ↓
Compute Reward (Reinforcement Learning)
       ↓
Update Model Weights
       ↓
Move Window: [Packet 11, Packet 12, ..., Packet 1010]
       ↓
Repeat...
```

## Key Components

### 1. PacketInfoPredictor Class
- **Model Management**: Handles GPT-2 model initialization and training
- **Prediction Engine**: Generates next k packet predictions
- **Reinforcement Learning**: Implements reward-based learning
- **Accuracy Calculation**: Bit-by-bit comparison for precise evaluation

### 2. Data Processing
- **Text to Binary Conversion**: Converts packet info to binary for bit-by-bit comparison
- **Padding**: Ensures uniform length for comparison
- **Tokenization**: Uses GPT-2 tokenizer for model input

### 3. Training Loop
- **Sliding Window**: Moves through data in chunks
- **Context Window**: Uses 1000 packets as context (configurable)
- **Prediction Window**: Predicts next 10 packets (configurable)
- **Iterative Learning**: Continuous improvement through RL

## Step-by-Step Implementation

### Step 1: Model Initialization
```python
predictor = PacketInfoPredictor(
    sequence_length=1000,    # Context window size
    prediction_length=10,    # Number of packets to predict
    learning_rate=1e-4       # Learning rate for optimization
)
```

### Step 2: Data Loading
```python
# Load PCAP data
packet_data = load_pcap_data("Pcap/pcap.csv")
# Extract 'Info' column (300,000 packets)
```

### Step 3: Training Process
For each iteration:
1. **Context Extraction**: Take packets [i, i+1000] as context
2. **Prediction**: Model predicts packets [i+1000, i+1010]
3. **Comparison**: Compare with actual packets [i+1000, i+1010]
4. **Bit Accuracy**: Convert to binary and compare bit-by-bit
5. **Reward Calculation**: Higher accuracy = higher reward
6. **Model Update**: Adjust weights based on reward
7. **Window Slide**: Move to next iteration [i+10, i+1010]

### Step 4: Accuracy Calculation
```python
def _calculate_bit_accuracy(self, predicted: str, actual: str, max_length: int) -> float:
    # Convert text to binary
    pred_binary = self._text_to_binary(predicted, max_length)
    actual_binary = self._text_to_binary(actual, max_length)
    
    # Compare bit by bit
    correct_bits = sum(1 for p, a in zip(pred_binary, actual_binary) if p == a)
    return correct_bits / len(pred_binary)
```

### Step 5: Reinforcement Learning
```python
def _calculate_reward(self, bit_accuracy: float) -> float:
    # Sigmoid-like reward function
    return 2.0 / (1.0 + np.exp(-10 * (bit_accuracy - 0.5))) - 1.0
```

## Usage

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run demo (small dataset)
python demo_predictor.py

# Run full training
python packet_predictor.py
```

### Configuration
```python
# Customize parameters
predictor = PacketInfoPredictor(
    sequence_length=500,     # Smaller context for faster training
    prediction_length=5,     # Predict fewer packets
    learning_rate=1e-3       # Higher learning rate
)
```

## Output and Monitoring

### 1. Training Progress
- **Accuracy**: Percentage of exact string matches
- **Bit Accuracy**: Bit-by-bit comparison accuracy
- **Reward**: Reinforcement learning reward value

### 2. Logging
- Console output for real-time monitoring
- File logging in `logs/packet_predictor.log`
- Training history saved as JSON

### 3. Visualization
- Training curves plotted automatically
- Saved as `training_history.png`
- Shows accuracy, bit accuracy, and reward over time

## Model Performance

### Expected Results
- **Initial Accuracy**: 0-10% (random predictions)
- **Final Accuracy**: 20-40% (after training)
- **Bit Accuracy**: Usually higher than string accuracy
- **Learning Curve**: Gradual improvement over iterations

### Factors Affecting Performance
1. **Data Quality**: Clean, consistent packet info
2. **Pattern Complexity**: Simple patterns learn faster
3. **Context Window**: Larger context = better predictions
4. **Training Iterations**: More iterations = better learning

## Technical Details

### Binary Conversion Process
```python
def _text_to_binary(self, text: str, max_length: int) -> str:
    # Convert each character to 8-bit binary
    binary = ''.join(format(ord(char), '08b') for char in text)
    
    # Pad to uniform length
    max_bits = max_length * 8
    binary = binary.ljust(max_bits, '0')
    
    return binary
```

### Model Architecture
- **Base Model**: GPT-2 (124M parameters)
- **Input**: Tokenized packet info sequences
- **Output**: Generated text predictions
- **Training**: Supervised learning with RL reward adjustment

### Memory Management
- **Batch Processing**: Processes data in chunks
- **Gradient Accumulation**: Handles large sequences
- **GPU Support**: Automatic CUDA detection and usage

## Future Improvements

1. **Custom Tokenizer**: Train tokenizer on packet-specific vocabulary
2. **Attention Visualization**: Understand what the model focuses on
3. **Multi-modal Input**: Include other packet fields (protocol, length, etc.)
4. **Ensemble Methods**: Combine multiple models for better predictions
5. **Real-time Prediction**: Stream processing for live network monitoring

## Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce sequence_length or batch_size
2. **Slow Training**: Use GPU or reduce model size
3. **Poor Accuracy**: Increase training iterations or adjust learning rate
4. **Import Errors**: Install all requirements with `pip install -r requirements.txt`

### Performance Tips
- Use GPU for faster training
- Start with demo script for testing
- Monitor memory usage with large datasets
- Adjust hyperparameters based on your data

## License

This project is for educational and research purposes. Please ensure compliance with your organization's data usage policies when working with network packet data.
