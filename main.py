from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, IntegerType, DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("Churn with PySpark") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df = spark.read.csv(r'bank_customer_churn_prediction.csv',header=True)

df.show(5)
"""
+-----------+------------+-------+------+---+------+---------+---------------+-----------+-------------+----------------+-----+
|customer_id|credit_score|country|gender|age|tenure|  balance|products_number|credit_card|active_member|estimated_salary|churn|
+-----------+------------+-------+------+---+------+---------+---------------+-----------+-------------+----------------+-----+
|   15634602|         619| France|Female| 42|     2|        0|              1|          1|            1|       101348.88|    1|
|   15647311|         608|  Spain|Female| 41|     1| 83807.86|              1|          0|            1|       112542.58|    0|
|   15619304|         502| France|Female| 42|     8| 159660.8|              3|          1|            0|       113931.57|    1|
|   15701354|         699| France|Female| 39|     1|        0|              2|          0|            0|        93826.63|    0|
|   15737888|         850|  Spain|Female| 43|     2|125510.82|              1|          1|            1|         79084.1|    0|
+-----------+------------+-------+------+---+------+---------+---------------+-----------+-------------+----------------+-----+
"""
# Size
print((df.count(), len(df.columns)))
"(10000, 12)"

df2 = df.filter(df["balance"] != 0).drop("customer_id")

df2.dtypes
"""
[('credit_score', 'string'),
 ('country', 'string'),
 ('gender', 'string'),
 ('age', 'string'),
 ('tenure', 'string'),
 ('balance', 'string'),
 ('products_number', 'string'),
 ('credit_card', 'string'),
 ('active_member', 'string'),
 ('estimated_salary', 'string'),
 ('churn', 'string')]
"""

columns_to_cast = {
    "credit_score": "int",
    "country": "string",
    "gender": "string",
    "age": "int",
    "tenure": "int",
    "balance": "double",
    "products_number": "int",
    "credit_card": "int",
    "active_member": "int",
    "estimated_salary": "int",
    "churn": "int"
}
for column, dtype in columns_to_cast.items():
    df2 = df2.withColumn(column, col(column).cast(dtype))

df2.dtypes
"""
[('credit_score', 'int'),
 ('country', 'string'),
 ('gender', 'string'),
 ('age', 'int'),
 ('tenure', 'int'),
 ('balance', 'double'),
 ('products_number', 'int'),
 ('credit_card', 'int'),
 ('active_member', 'int'),
 ('estimated_salary', 'int'),
 ('churn', 'int')]
"""

cat_cols = [cl for cl in df2.columns if df2.schema[cl].dataType == StringType()]
num_cols = [cl for cl in df2.columns if df2.schema[cl].dataType in [IntegerType(), DoubleType()] and cl != "churn"]

for col_name in cat_cols:
    print(f"{col_name} : {df2.groupBy(col_name).count().show()}")
"""
+-------+-----+
|country|count|
+-------+-----+
|Germany| 2509|
| France| 2596|
|  Spain| 1278|
+-------+-----+
+------+-----+
|gender|count|
+------+-----+
|Female| 2889|
|  Male| 3494|
+------+-----+
"""

# OUTLIERS
threshold = 3
for col_name in num_cols:
    mean_val = df2.select(mean(col(col_name))).collect()[0][0]
    stddev_val = df2.select(stddev(col(col_name))).collect()[0][0]

    lower_threshold = mean_val - threshold * stddev_val
    upper_threshold = mean_val + threshold * stddev_val
    print(f"{col_name} : {df2.filter((col(col_name) < lower_threshold) | (col(col_name) > upper_threshold)).count()}")
"""
credit_score : 6
age : 78
tenure : 0
balance : 24
products_number : 46
credit_card : 0
active_member : 0
estimated_salary : 0
"""

def fix_outliers(df, num_cols, threshold):
    for col_name in num_cols:
        mean_val = df.select(mean(col(col_name))).collect()[0][0]
        stddev_val = df.select(stddev(col(col_name))).collect()[0][0]

        lower_threshold = mean_val - threshold * stddev_val
        upper_threshold = mean_val + threshold * stddev_val

        df = df.filter((col(col_name) >= lower_threshold) & (col(col_name) <= upper_threshold))

    return df

df2 = fix_outliers(df2, num_cols, 3)

# Label Encoding
index_cols = ["countryIndex", "genderIndex"]
indexer = StringIndexer(inputCols=["country", "gender"], outputCols=index_cols)
indexed_df = indexer.fit(df2).transform(df2)
indexed_df.show(5)
"""
+------------+-------+------+---+------+---------+---------------+-----------+-------------+----------------+-----+------------+-----------+
|credit_score|country|gender|age|tenure|  balance|products_number|credit_card|active_member|estimated_salary|churn|countryIndex|genderIndex|
+------------+-------+------+---+------+---------+---------------+-----------+-------------+----------------+-----+------------+-----------+
|         608|  Spain|Female| 41|     1| 83807.86|              1|          0|            1|          112542|    0|         2.0|        1.0|
|         850|  Spain|Female| 43|     2|125510.82|              1|          1|            1|           79084|    0|         2.0|        1.0|
|         645|  Spain|  Male| 44|     8|113755.78|              2|          1|            0|          149756|    1|         2.0|        0.0|
|         501| France|  Male| 44|     4|142051.07|              2|          0|            1|           74940|    0|         0.0|        0.0|
|         684| France|  Male| 27|     2|134603.88|              1|          1|            1|           71725|    0|         0.0|        0.0|
+------------+-------+------+---+------+---------+---------------+-----------+-------------+----------------+-----+------------+-----------+
"""

# Scaling
assem = VectorAssembler(inputCols=['credit_score','countryIndex','genderIndex','age','tenure','balance','products_number','credit_card','active_member','estimated_salary'], outputCol='features')
assem_data = assem.transform(indexed_df)

scaler = StandardScaler(inputCol='features', outputCol='standardized_features', withMean=True, withStd=True)
scaled_data = scaler.fit(assem_data).transform(assem_data)
scaled_data.select("standardized_features", "churn").show(5)
"""
+---------------------+-----+
|standardized_features|churn|
+---------------------+-----+
| [-0.4479797088076...|    0|
| [2.06187000712732...|    0|
| [-0.0642423555448...|    1|
| [-1.5577066493243...|    0|
| [0.34023755735375...|    0|
+---------------------+-----+
"""
final_df = scaled_data.select("standardized_features", "churn")

# Train and test split
train, test = final_df.randomSplit([0.8, 0.2], seed=21)

# Modelling
model = GBTClassifier(featuresCol = "standardized_features", labelCol = "churn")
model = model.fit(train)

# Predictions with test dataset
predictions = model.transform(test)

# Score Metrics
evaluator = MulticlassClassificationEvaluator(labelCol="churn", predictionCol="prediction")
metrics = {metric: evaluator.evaluate(predictions, {evaluator.metricName: metric})
           for metric in ["accuracy", "weightedRecall", "f1"]}
roc_auc = BinaryClassificationEvaluator(labelCol='churn', rawPredictionCol='prediction', metricName='areaUnderROC').evaluate(predictions)
