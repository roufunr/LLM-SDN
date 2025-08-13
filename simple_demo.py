#!/usr/bin/env python3
"""
Simple Demo: Packet Info Predictor Concept

This script demonstrates the packet prediction concept without requiring
heavy ML dependencies. It shows the data flow and evaluation process.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import json
from pathlib import Path


class SimplePacketPredictor:
    """Simplified packet predictor for demonstration."""
    
    def __init__(self, sequence_length: int = 100, prediction_length: int = 5):
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.training_history = {
            'accuracies': [],
            'bit_accuracies': [],
            'iterations': []
        }
    
    def _text_to_binary(self, text: str, max_length: int) -> str:
        """Convert text to binary representation with padding."""
        # Convert text to binary
        binary = ''.join(format(ord(char), '08b') for char in text)
        
        # Pad to max_length * 8 bits
        max_bits = max_length * 8
        if len(binary) < max_bits:
            binary = binary.ljust(max_bits, '0')
        else:
            binary = binary[:max_bits]
        
        return binary
    
    def _calculate_bit_accuracy(self, predicted: str, actual: str, max_length: int) -> float:
        """Calculate bit-by-bit accuracy between predicted and actual text."""
        pred_binary = self._text_to_binary(predicted, max_length)
        actual_binary = self._text_to_binary(actual, max_length)
        
        if len(pred_binary) != len(actual_binary):
            min_length = min(len(pred_binary), len(actual_binary))
            pred_binary = pred_binary[:min_length]
            actual_binary = actual_binary[:min_length]
        
        correct_bits = sum(1 for p, a in zip(pred_binary, actual_binary) if p == a)
        total_bits = len(pred_binary)
        
        return correct_bits / total_bits if total_bits > 0 else 0.0
    
    def predict_next_packets_simple(self, context_packets: List[str]) -> List[str]:
        """
        Simple prediction based on pattern matching.
        In a real implementation, this would be replaced by a transformer model.
        """
        # This is a simplified prediction that looks for patterns
        # In reality, this would be the transformer model's output
        
        # Analyze context for common patterns
        protocols = {}
        for packet in context_packets[-10:]:  # Look at last 10 packets
            # Ensure packet is a string
            packet_str = str(packet) if packet is not None else ""
            
            if 'TCP' in packet_str:
                protocols['TCP'] = protocols.get('TCP', 0) + 1
            elif 'UDP' in packet_str:
                protocols['UDP'] = protocols.get('UDP', 0) + 1
            elif 'ICMP' in packet_str:
                protocols['ICMP'] = protocols.get('ICMP', 0) + 1
            elif 'DNS' in packet_str:
                protocols['DNS'] = protocols.get('DNS', 0) + 1
            elif 'HTTP' in packet_str:
                protocols['HTTP'] = protocols.get('HTTP', 0) + 1
        
        # Predict based on most common protocol
        most_common = max(protocols.items(), key=lambda x: x[1])[0] if protocols else 'TCP'
        
        # Generate predictions based on pattern
        predictions = []
        for i in range(self.prediction_length):
            if most_common == 'TCP':
                predictions.append(f"TCP {1000+i} > {2000+i} [ACK] Seq={1000+i} Ack={2000+i} Win=65535 Len={i}")
            elif most_common == 'UDP':
                predictions.append(f"UDP {1000+i} > {2000+i} Len={50+i}")
            elif most_common == 'ICMP':
                predictions.append(f"ICMP Echo (ping) request id=0x01b5, seq={i}/256, ttl=64")
            elif most_common == 'DNS':
                predictions.append(f"DNS Standard query 0x{1000+i:04x} A example{i}.com")
            elif most_common == 'HTTP':
                predictions.append(f"HTTP GET /api/data{i} HTTP/1.1")
            else:
                predictions.append(f"Generic packet info {i}")
        
        return predictions
    
    def evaluate_prediction(self, predicted: List[str], actual: List[str]) -> Tuple[float, float]:
        """Evaluate prediction accuracy."""
        # Ensure all values are strings
        predicted = [str(p) for p in predicted]
        actual = [str(a) for a in actual]
        
        # Calculate exact match accuracy
        exact_matches = sum(1 for pred, act in zip(predicted, actual) 
                           if pred.strip() == act.strip())
        accuracy = exact_matches / len(actual)
        
        # Calculate bit accuracy
        max_length = max(len(info) for info in actual + predicted)
        bit_accuracies = []
        for pred, act in zip(predicted, actual):
            bit_acc = self._calculate_bit_accuracy(pred, act, max_length)
            bit_accuracies.append(bit_acc)
        
        avg_bit_accuracy = np.mean(bit_accuracies)
        
        return accuracy, avg_bit_accuracy
    
    def run_simulation(self, packet_data: List[str], num_iterations: int = 20):
        """Run the prediction simulation."""
        print(f"Running simulation with {len(packet_data)} packets")
        print(f"Sequence length: {self.sequence_length}, Prediction length: {self.prediction_length}")
        print("=" * 60)
        
        for iteration in range(num_iterations):
            # Calculate indices
            start_idx = iteration * self.prediction_length
            end_idx = start_idx + self.sequence_length
            pred_start = end_idx
            pred_end = pred_start + self.prediction_length
            
            # Check if we have enough data
            if pred_end > len(packet_data):
                print(f"Reached end of data at iteration {iteration}")
                break
            
            # Get context and target
            context_packets = packet_data[start_idx:end_idx]
            actual_next_packets = packet_data[pred_start:pred_end]
            
            # Make prediction
            predicted_packets = self.predict_next_packets_simple(context_packets)
            
            # Evaluate
            accuracy, bit_accuracy = self.evaluate_prediction(predicted_packets, actual_next_packets)
            
            # Store results
            self.training_history['iterations'].append(iteration)
            self.training_history['accuracies'].append(accuracy)
            self.training_history['bit_accuracies'].append(bit_accuracy)
            
            # Print results
            print(f"Iteration {iteration:2d}: Accuracy={accuracy:6.2%}, Bit Accuracy={bit_accuracy:6.2%}")
            
            # Show sample predictions every 5 iterations
            if iteration % 5 == 0:
                print("  Sample predictions:")
                for i, (pred, actual) in enumerate(zip(predicted_packets[:3], actual_next_packets[:3])):
                    print(f"    Pred {i+1}: {pred[:50]}...")
                    print(f"    Actual {i+1}: {actual[:50]}...")
                print()
        
        # Print summary
        print("=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        print(f"Total iterations: {len(self.training_history['iterations'])}")
        print(f"Average accuracy: {np.mean(self.training_history['accuracies']):.2%}")
        print(f"Average bit accuracy: {np.mean(self.training_history['bit_accuracies']):.2%}")
        print(f"Best accuracy: {max(self.training_history['accuracies']):.2%}")
        print(f"Best bit accuracy: {max(self.training_history['bit_accuracies']):.2%}")


def load_sample_data():
    """Load a sample of the PCAP data."""
    print("Loading sample PCAP data...")
    
    try:
        # Load first 1000 rows for demo
        df = pd.read_csv("Pcap/pcap.csv", nrows=1000)
        # Handle NaN values and convert to strings
        packet_info = df['Info'].fillna("").astype(str).tolist()
        print(f"Loaded {len(packet_info)} packets")
        return packet_info
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating synthetic data for demo...")
        
        # Create synthetic data based on the patterns we saw
        synthetic_data = []
        protocols = ['TCP', 'UDP', 'ICMP', 'DNS', 'HTTP', 'ARP']
        
        for i in range(1000):
            protocol = protocols[i % len(protocols)]
            if protocol == 'TCP':
                info = f"TCP {1000 + i} > {2000 + i} [ACK] Seq={1000 + i} Ack={2000 + i} Win=65535 Len={i % 100}"
            elif protocol == 'UDP':
                info = f"UDP {1000 + i} > {2000 + i} Len={50 + i % 50}"
            elif protocol == 'ICMP':
                info = f"ICMP Echo (ping) request id=0x01b5, seq={i}/256, ttl=64"
            elif protocol == 'DNS':
                info = f"DNS Standard query 0x{1000 + i:04x} A example{i}.com"
            elif protocol == 'HTTP':
                info = f"HTTP GET /api/data{i} HTTP/1.1"
            else:
                info = f"ARP Who has 192.168.1.{i % 255}? Tell 192.168.1.1"
            
            synthetic_data.append(info)
        
        print(f"Created {len(synthetic_data)} synthetic packets")
        return synthetic_data


def main():
    """Main demo function."""
    print("Simple Packet Info Predictor Demo")
    print("=" * 50)
    print("This demo shows the concept without requiring heavy ML dependencies.")
    print("In a real implementation, the simple prediction would be replaced")
    print("by a transformer model (GPT-2) with reinforcement learning.")
    print()
    
    # Load data
    packet_data = load_sample_data()
    
    # Initialize predictor
    predictor = SimplePacketPredictor(
        sequence_length=100,  # Context window
        prediction_length=5   # Predict 5 packets
    )
    
    # Run simulation
    predictor.run_simulation(packet_data, num_iterations=20)
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'simple_demo_results.json', 'w') as f:
        json.dump(predictor.training_history, f, indent=2)
    
    print(f"\nResults saved to {results_dir / 'simple_demo_results.json'}")
    print("\nTo run the full transformer-based system:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run: python3 demo_predictor.py")


if __name__ == "__main__":
    main()
