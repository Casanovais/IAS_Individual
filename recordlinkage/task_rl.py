"""
Task Manager for Privacy Risk Evaluation (Record Linkage)
"""
import sys
import glob
import argparse
import pika

parser = argparse.ArgumentParser(description='Privacy Task Manager')
parser.add_argument('--input_folder', type=str, help='Input folder', default="./input")
args = parser.parse_args()

# Connect to RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

# CRITICAL: Must match the queue name in worker_rl.py ('task_rl')
channel.queue_declare(queue='task_rl', durable=True, arguments={"dead-letter-exchange":"dlx"})

# Find all CSV files in the input folder
files = glob.glob(f"{args.input_folder}/*.csv")

print(f" [*] Sending {len(files)} tasks to queue 'task_rl'...")

for file_path in files:
    # Extract filename only (e.g., 'adult_knn1.csv')
    file_name = file_path.split('/')[-1]
    
    message = file_name
    
    # Publish the message
    channel.basic_publish(
        exchange='',
        routing_key='task_rl',  # Target the specific privacy queue
        body=message,
        properties=pika.BasicProperties(
            delivery_mode=2,  # Make message persistent
        ))
    print(f" [x] Sent {file_name}")

connection.close()