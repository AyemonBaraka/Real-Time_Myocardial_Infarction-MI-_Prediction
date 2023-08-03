from kafka import KafkaProducer
import csv
from time import sleep
import json
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# Kafka broker address
bootstrap_servers = '10.18.17.153:9092'
# Kafka topics to send the data
topic1 = 'ecg_data_normal'
topic2 = 'ecg_data_abnormal'

# Create a Kafka producer
producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))
# CSV file paths
csv_file1 = '/home/ayemon/KafkaProjects/kafkaspark07_90/ptbdb_normal_20.csv'
csv_file2 = '/home/ayemon/KafkaProjects/kafkaspark07_90/ptbdb_abnormal_20.csv'

# Function to read a CSV file and send messages to Kafka with label
def send_csv_to_kafka_with_label(csv_file, topic):
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Get the header
        for row in csv_reader:
            # Assuming each row is a list of numerical values
            message = dict(zip(header, row))
            producer.send(topic, value=message)
            print(f"Sending data to topic '{topic}': {message}")
            sleep(2)

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(send_csv_to_kafka_with_label, csv_file1, topic1)
        future2 = executor.submit(send_csv_to_kafka_with_label, csv_file2, topic2)

        # Wait for both tasks to complete
        concurrent.futures.wait([future1, future2])

    producer.close()