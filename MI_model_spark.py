from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType,StructField,LongType, StringType,DoubleType,TimestampType
from pyspark.ml.feature import MinMaxScaler, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


topic1 = "ecg_data_normal"
topic2 = "ecg_data_abnormal"



# Create Spark session
spark = SparkSession.builder \
    .appName("Spark_ECG_model") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Define the schema for the CSV data (assuming 188 columns as in previous examples)
#schema = StructType([StructField("t" + str(i+1), DoubleType(), True) for i in range(188)])

schema = StructType([
    StructField("t" + str(i+1), DoubleType(), True) for i in range(187)
] + [
    StructField("label", LongType(), True)
])

# Load the CSV files into DataFrames
df1 = spark.read.format('csv').option("header", "true").schema(schema).load("/home/ayemon/KafkaProjects/kafkaspark07_90/ptbdb_normal_80_label.csv")
#df1.printSchema()
#df1.show()

df2 = spark.read.format('csv').option("header", "true").schema(schema).load("/home/ayemon/KafkaProjects/kafkaspark07_90/ptbdb_abnormal_80_label.csv")
#df2.printSchema()

df = df1.union(df2)
#df.printSchema()


trainDF, testDF = df.randomSplit([0.75, 0.25], seed=42)

feature_cols = df.columns
lr = LogisticRegression(maxIter=1000, regParam= 0.01)
assembler1 = VectorAssembler(inputCols=feature_cols, outputCol="features_scaled1")
scaler = MinMaxScaler(inputCol="features_scaled1", outputCol="features_scaled2")
assembler2 = VectorAssembler(inputCols=['features_scaled2'], outputCol="features")
myStages = [assembler1, scaler, assembler2,lr]
pipeline = Pipeline(stages= myStages)


# We fit the model using the training data.
pModel = pipeline.fit(trainDF)# We transform the data.
trainingPred = pModel.transform(testDF)# # We select the actual label, probability and predictions
trainingPred.select('label','probability','prediction').show()

# Evaluate the model's performance
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator.evaluate(trainingPred)
print("Accuracy: ", accuracy)

trainingPred.crosstab('label','prediction').show()

# Save the trained model
model_save_path = "/home/ayemon/KafkaProjects/kafkaspark07_90/model_ecg"
pModel.write().overwrite().save(model_save_path)



