from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, LongType, DoubleType
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from pyspark.sql.functions import col

# Create Spark session
spark = SparkSession.builder \
    .appName("Spark_ECG_MLP_Model") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Define the schema for the CSV data
schema = StructType([
    StructField("t" + str(i+1), DoubleType(), True) for i in range(187)
] + [
    StructField("label", LongType(), True)
])

# Load the CSV files into DataFrames
df1 = spark.read.format('csv').option("header", "true").schema(schema).load("/home/ayemon/KafkaProjects/kafkaspark07_90/ptbdb_normal_80_label.csv")
df2 = spark.read.format('csv').option("header", "true").schema(schema).load("/home/ayemon/KafkaProjects/kafkaspark07_90/ptbdb_abnormal_80_label.csv")

# Combine both datasets
df = df1.union(df2)

# Split data into training and testing datasets
trainDF, testDF = df.randomSplit([0.75, 0.25], seed=42)

# Define feature columns
feature_cols = df.columns[:-1]  # Exclude the label column

# Define MLP Model
# Layers: Input size (number of features), hidden layers, output size (number of classes)
layers = [187, 128, 64, 32, 2]  # Adjust based on your dataset's features and classes

mlp = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", layers=layers, seed=42, maxIter=200, blockSize=128)

# Define the pipeline stages
assembler1 = VectorAssembler(inputCols=feature_cols, outputCol="features_scaled1")
scaler = MinMaxScaler(inputCol="features_scaled1", outputCol="features_scaled2")
assembler2 = VectorAssembler(inputCols=['features_scaled2'], outputCol="features")
pipeline = Pipeline(stages=[assembler1, scaler, assembler2, mlp])

# Train the model
print("Training the MLP model...")
pModel = pipeline.fit(trainDF)

# Predict on the training data
trainingPred = pModel.transform(trainDF)
trainingPred.select('label', 'prediction', 'probability').show()

# Evaluate on the training data
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
training_accuracy = evaluator.evaluate(trainingPred)
print(f"Training Accuracy: {training_accuracy}")

# Predict on the test data
testPred = pModel.transform(testDF)

# Evaluate the model's performance on the test data
test_accuracy = evaluator.evaluate(testPred)
print(f"Test Accuracy: {test_accuracy}")

# Print evaluation metrics
predictions_and_labels = testPred.select(col("prediction"), col("label")).rdd.map(lambda x: (float(x[0]), float(x[1])))
metrics = MulticlassMetrics(predictions_and_labels)

# Confusion matrix
confusion_matrix = metrics.confusionMatrix().toArray()
true_positive = confusion_matrix[1, 1]
false_positive = confusion_matrix[0, 1]
true_negative = confusion_matrix[0, 0]
false_negative = confusion_matrix[1, 0]

# Sensitivity (Recall)
sensitivity = true_positive / (true_positive + false_negative)

# Specificity
specificity = true_negative / (true_negative + false_positive)

# Precision
precision = true_positive / (true_positive + false_positive)

# F1 Score
f1_score = metrics.fMeasure(1.0)

# Calculate ROC AUC
# Extract probability and label columns for BinaryClassificationMetrics
probability_and_labels = testPred.select(col("probability").alias("score"), col("label")).rdd.map(lambda row: (float(row.score[1]), float(row.label)))
binary_metrics = BinaryClassificationMetrics(probability_and_labels)

roc_auc = binary_metrics.areaUnderROC

# Print metrics
print("Confusion Matrix:")
print(confusion_matrix)
print(f"Test Accuracy: {test_accuracy}")
print(f"Sensitivity (Recall): {sensitivity}")
print(f"Specificity: {specificity}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1_score}")
print(f"ROC AUC: {roc_auc}")

# Save the trained model
model_save_path = "/home/ayemon/KafkaProjects/kafkaspark07_90/mlp_model"
pModel.write().overwrite().save(model_save_path)

print(f"MLP model saved at {model_save_path}")
