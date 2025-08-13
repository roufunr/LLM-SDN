#!/usr/bin/env python3
"""
Demo script for Packet Info Predictor

This script demonstrates the packet prediction system with a smaller dataset
for testing and understanding the workflow.
"""

import pandas as pd
import numpy as np
from packet_predictor import PacketInfoPredictor, load_pcap_data
import logging

def create_demo_data():
    """Create a small demo dataset for testing."""
    print("Creating demo dataset...")
    
    # Sample packet info patterns based on the actual PCAP data
    demo_info = [
        "TCP 40234 > 5228 [PSH, ACK] Seq=1 Ack=1 Win=65360 Len=4",
        "TCP [TCP Dup ACK 1#1] 40234 > 5228 [ACK] Seq=5 Ack=1 Win=65360",
        "TCP 443 > 46330 [ACK] Seq=1 Ack=1 Win=963 Len=0",
        "TCP 3063 > 443 [ACK] Seq=1 Ack=1 Win=3072 Len=1",
        "ICMP Echo (ping) reply id=0x01b5, seq=1/256, ttl=45",
        "DNS Standard query 0x289b PTR 173.199.154.195.in-addr.arpa",
        "TCP 5228 > 40234 [PSH, ACK] Seq=1 Ack=5 Win=63063 Len=2",
        "TCP 443 > 3063 [ACK] Seq=1 Ack=2 Win=19832 Len=0",
        "ARP Who has 192.168.1.1? Tell 192.168.1.223",
        "ARP 192.168.1.1 is at 14:cc:20:51:33:ea",
        "UDPENCAP NAT-keepalive",
        "DNS Standard query response 0x289b PTR 173.199.154.195.in-addr.arpa",
        "ICMP Echo (ping) request id=0x01b5, seq=2/512, ttl=64",
        "TLSv1 Application Data, Application Data",
        "ICMPv6 Multicast Listener Report",
        "ICMP Echo (ping) reply id=0x01b5, seq=2/512, ttl=45",
        "TCP [TCP segment of a reassembled PDU]",
        "HTTP GET /api/data HTTP/1.1",
        "HTTP 200 OK",
        "TLSv1.2 Client Hello",
        "TLSv1.2 Server Hello",
        "TLSv1.2 Certificate, Server Key Exchange, Server Hello Done",
        "DHCP Discover",
        "DHCP Offer",
        "DHCP Request",
        "DHCP ACK",
        "NTP Version 3, server",
        "NTP Version 4, server",
        "SSDP M-SEARCH * HTTP/1.1",
        "SSDP HTTP/1.1 200 OK",
        "EAPOL EAPOL-Key (1/4)",
        "EAPOL EAPOL-Key (2/4)",
        "EAPOL EAPOL-Key (3/4)",
        "EAPOL EAPOL-Key (4/4)",
        "IGMPv2 Membership Report",
        "ESP Encrypted Payload",
        "STUN Binding Request",
        "STUN Binding Response",
        "SIP INVITE sip:user@example.com SIP/2.0",
        "SIP 200 OK",
        "SIP ACK sip:user@example.com SIP/2.0",
        "HTTP/XML POST /soap HTTP/1.1",
        "HTTP/XML HTTP/1.1 200 OK"
    ]
    
    # Create a larger dataset by repeating and varying the patterns
    full_dataset = []
    for i in range(1000):  # Create 1000 packets
        base_info = demo_info[i % len(demo_info)]
        # Add some variation
        if "TCP" in base_info:
            # Vary port numbers and sequence numbers
            import random
            port1 = random.randint(1000, 65535)
            port2 = random.randint(1000, 65535)
            seq = random.randint(1, 1000000)
            ack = random.randint(1, 1000000)
            win = random.randint(1000, 65535)
            len_val = random.randint(0, 1500)
            
            # Replace values in the string
            modified = base_info.replace("40234", str(port1))
            modified = modified.replace("5228", str(port2))
            modified = modified.replace("Seq=1", f"Seq={seq}")
            modified = modified.replace("Ack=1", f"Ack={ack}")
            modified = modified.replace("Win=65360", f"Win={win}")
            modified = modified.replace("Len=4", f"Len={len_val}")
            
            full_dataset.append(modified)
        else:
            full_dataset.append(base_info)
    
    print(f"Created demo dataset with {len(full_dataset)} packets")
    return full_dataset

def run_demo():
    """Run the demo with a small dataset."""
    print("Packet Info Predictor Demo")
    print("=" * 40)
    
    # Create demo data
    demo_data = create_demo_data()
    
    # Initialize predictor with smaller parameters for demo
    predictor = PacketInfoPredictor(
        sequence_length=100,  # Smaller context window
        prediction_length=5,   # Predict 5 packets at a time
        learning_rate=1e-3     # Slightly higher learning rate
    )
    
    print(f"Model initialized with:")
    print(f"  Sequence length: {predictor.sequence_length}")
    print(f"  Prediction length: {predictor.prediction_length}")
    print(f"  Learning rate: {predictor.learning_rate}")
    print(f"  Device: {predictor.device}")
    
    # Run training with fewer iterations for demo
    print("\nStarting training...")
    predictor.run_training(demo_data, num_iterations=20)
    
    # Test prediction on a new sequence
    print("\nTesting prediction on new sequence...")
    test_context = demo_data[200:300]  # Use packets 200-299 as context
    test_target = demo_data[300:305]   # Use packets 300-304 as target
    
    print("Context (first 3 packets):")
    for i, packet in enumerate(test_context[:3]):
        print(f"  {i+1}: {packet}")
    
    print("\nActual next 5 packets:")
    for i, packet in enumerate(test_target):
        print(f"  {i+1}: {packet}")
    
    # Make prediction
    predicted = predictor.predict_next_packets(test_context)
    
    print("\nPredicted next 5 packets:")
    for i, packet in enumerate(predicted):
        print(f"  {i+1}: {packet}")
    
    # Calculate accuracy
    exact_matches = sum(1 for pred, actual in zip(predicted, test_target) 
                       if pred.strip() == actual.strip())
    accuracy = exact_matches / len(test_target)
    
    print(f"\nExact match accuracy: {accuracy:.2%}")
    
    # Save model
    predictor.save_model("demo_model")
    print("\nDemo completed! Model saved to 'demo_model' directory.")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        run_demo()
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
