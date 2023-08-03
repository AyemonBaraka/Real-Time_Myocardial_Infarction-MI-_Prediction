from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, LongType
from pyspark.sql.functions import col, from_json
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load the saved logistic regression model
model_save_path = "/home/ayemon/KafkaProjects/kafkaspark07_90/model_ecg"  # Replace with the path where you saved the model
loaded_lrModel = PipelineModel.load(model_save_path)

# Kafka broker address
kafka_broker = "10.18.17.153:9092"
# Kafka topics to read from
topic1 = "ecg_data_normal"
topic2 = "ecg_data_abnormal"

# Create Spark session
spark = SparkSession.builder \
    .appName("KafkaSparkConsumer") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Define the schema for the CSV data (assuming 188 columns as in previous examples)
#schema = StructType([StructField("t" + str(i+1), StringType(), True) for i in range(188)])

schema = StructType([
    StructField("t" + str(i+1), StringType(), True) for i in range(187)
] + [
    StructField("label", StringType(), True)
])

# Define the Kafka consumer settings
kafka_df1 = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_broker) \
    .option("subscribe", topic1) \
    .load()

kafka_df2 = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_broker) \
    .option("subscribe", topic2) \
    .load()

# Convert the Kafka value to a DataFrame column
kafka_df1 = kafka_df1.selectExpr("CAST(value AS STRING)")
kafka_df2 = kafka_df2.selectExpr("CAST(value AS STRING)")


# Apply the schema to the data
kafka_df1 = kafka_df1.select(from_json("value", schema).alias("data")).select("data.*")
kafka_df2 = kafka_df2.select(from_json("value", schema).alias("data")).select("data.*")


# Change the data types of columns to DoubleType
for i in range(187):
    kafka_df1 = kafka_df1.withColumn("t" + str(i+1), col("t" + str(i+1)).cast("double"))
kafka_df1 = kafka_df1.withColumn("label", col("label").cast("long"))

for i in range(187):
    kafka_df2 = kafka_df2.withColumn("t" + str(i+1), col("t" + str(i+1)).cast("double"))
kafka_df2 = kafka_df2.withColumn("label", col("label").cast("long"))

#kafka_df1.printSchema()
#kafka_df2.printSchema()


predictions1 = loaded_lrModel.transform(kafka_df1)
predictions1 = predictions1.select('label','prediction')

predictions2 = loaded_lrModel.transform(kafka_df2)
predictions2 = predictions2.select('label','prediction')

# Start the streaming query to continuously read from Kafka and make predictions
query1 = predictions1.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query2 = predictions2.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query1.awaitTermination()
query2.awaitTermination()
