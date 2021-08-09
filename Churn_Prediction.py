import warnings

import findspark
import pandas as pd
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import functions as sf
from pyspark.sql import SparkSession

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

findspark.init("/Users/aslihankalyonkat/spark/spark-3.1.2-bin-hadoop3.2")

# Creating spark session
spark = SparkSession.builder \
    .master("local") \
    .appName("churn_prediction") \
    .getOrCreate()

# Get spark context
sc = spark.sparkContext

################################################    OBJECTIVE    ################################################
# It is expected to develop a machine learning model that can predict customers who will leave the company.
#################################################################################################################

#################################################  ABOUT DATASET  ###############################################
# Consists of 10000 observations and 12 variables.
# The independent variables contain information about customers.
# The dependent variable represents the customer abandonment status.
#################################################################################################################

################################################   DEĞİŞKENLER   ################################################
# Surname         – Customer surname
# CreditScore     – Customer's credit score
# Geography       – Country where the customer is located
# Gender          – Customer's gender
# Age             – Customer's age
# Tenure          – Information on how many years of customer it is
# NumOfProducts   – Used bank product
# HasCrCard       – Credit card status (0=No,1=Yes)
# IsActiveMember  – Active Membership status (0=No,1=Yes)
# EstimatedSalary – Customer's estimated salary
# Exited:         – Exited or not (0=No,1=Yes)
#################################################################################################################

# Reading dataset
df = spark.read.csv("datasets/churn.csv", header=True, inferSchema=True)

df.show()
print("Shape: ", (df.count(), len(df.columns)))
# Shape:  (10000, 14)

#####################################################
# Exploratory Data Analysis
#####################################################

# Variable types
df.printSchema()

# Lowering column names
df = df.toDF(*[c.lower() for c in df.columns])

# avarage balance, estimatedsalary and age by exited
df.groupby("exited").agg({"age": "mean", "balance": "mean", "estimatedsalary": "mean"}).show()
# +------+-----------------+--------------------+-----------------+
# |exited|     avg(balance)|avg(estimatedsalary)|         avg(age)|
# +------+-----------------+--------------------+-----------------+
# |     1|91108.53933726063|   101465.6775306824| 44.8379970544919|
# |     0|72745.29677885193|   99738.39177194514|37.40838879819164|
# +------+-----------------+--------------------+-----------------+

# Exited percentage by gender, geography, numofproducts
df.groupby(["gender", "geography", "numofproducts"]).agg({"exited": "mean"}).show(24)
# For case numofproducts = 4, customers has never exited

# Dropping useless columns
df = df.drop('surname', 'rownumber')

# Function for getting categorical and numeric columns
def get_col_types(dt):
    # Selection of all numeric columns
    num_cols = [col[0] for col in df.dtypes if col[1] != 'string']

    # Numeric but categorical columns
    num_but_cat_cols = []
    for col in num_cols:
        if (df.select(col).distinct().count() < 10):
            num_but_cat_cols.append(col)

    # Selection of all categorical columns
    cat_cols = [col[0] for col in df.dtypes if col[1] == 'string']

    # Adding the num_but_cat_cols list to cat_cols and deleting it from num_cols.
    cat_cols += num_but_cat_cols
    num_cols = [col for col in num_cols if col not in num_but_cat_cols]

    return num_cols, cat_cols


num_cols, cat_cols = get_col_types(df)

# unique values for all categorical columns
for col in cat_cols:
    df.select(col).distinct().show()

#####################################################
# DATA PREPROCESSING & FEATURE ENGINEERING
#####################################################

##### Missing Values #####
from pyspark.sql.functions import when, count, col, array

df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).toPandas().T
# no missing values

##### Feature Engineering #####
# Age Categories
df = df.withColumn('age_cat',
                   when(df['age'] < 36, "young").
                   when((35 < df['age']) & (df['age'] < 46), "mature").
                   otherwise("senior"))
df.groupby("age_cat", "exited").count().show()
# +-------+------+-----+
# |age_cat|exited|count|
# +-------+------+-----+
# | senior|     1|  957|
# |  young|     1|  347|
# |  young|     0| 3806|
# | senior|     0| 1154|
# | mature|     1|  733|
# | mature|     0| 3003|
# +-------+------+-----+

# Tenure Categories
df = df.withColumn('tenure_cat',
                   when(df['tenure'] < 4, "new_customer").
                   when((3 < df['tenure']) & (df['tenure'] < 7), "regular_customer").
                   when((6 < df['tenure']) & (df['tenure'] < 10), "loyal_customer").
                   otherwise("champion"))
df.groupby("tenure_cat", "exited").count().show()
# +----------------+------+-----+
# |      tenure_cat|exited|count|
# +----------------+------+-----+
# |    new_customer|     0| 2764|
# |    new_customer|     1|  741|
# |  loyal_customer|     0| 2450|
# |        champion|     1|  101|
# |  loyal_customer|     1|  587|
# |regular_customer|     1|  608|
# |regular_customer|     0| 2360|
# |        champion|     0|  389|
# +----------------+------+-----+

# Age and gender categories
df = df.withColumn('age_gender_cat', sf.concat(sf.col('age_cat'),sf.lit('_'), sf.col('gender')))
df.groupby("age_gender_cat", "exited").count().show()
# +--------------+------+-----+
# |age_gender_cat|exited|count|
# +--------------+------+-----+
# |   senior_Male|     1|  422|
# | mature_Female|     0| 1275|
# |    young_Male|     1|  144|
# |    young_Male|     0| 2177|
# |   mature_Male|     1|  332|
# | senior_Female|     1|  535|
# |  young_Female|     1|  203|
# |   senior_Male|     0|  654|
# |  young_Female|     0| 1629|
# | senior_Female|     0|  500|
# |   mature_Male|     0| 1728|
# | mature_Female|     1|  401|
# +--------------+------+-----+

# Kaç yaşında müşterimiz oldu ?
df = df.withColumn('age_that_become_customer', df.age - df.tenure)
df.groupby("exited").agg({'age_that_become_customer': "mean"}).show()
# +------+-----------------------------+
# |exited|avg(age_that_become_customer)|
# +------+-----------------------------+
# |     1|            39.90525282277859|
# |     0|           32.375109883209845|
# +------+-----------------------------+

# Hesap miktari / Maaş oranı
df = df.withColumn('balance_salary_ratio', df.balance / df.estimatedsalary)
df.groupby("exited").agg({'balance_salary_ratio': "mean"}).show()
# +------+-------------------------+
# |exited|avg(balance_salary_ratio)|
# +------+-------------------------+
# |     1|         9.35297659213245|
# |     0|       2.4783393189456544|
# +------+-------------------------+

# has credit card or active member
df = df.withColumn('hascrcard_or_activemember', df.hascrcard * df.isactivemember)
df.groupby("hascrcard_or_activemember", "exited").count().show()
# +-------------------------+------+-----+
# |hascrcard_or_activemember|exited|count|
# +-------------------------+------+-----+
# |                        1|     0| 3125|
# |                        1|     1|  482|
# |                        0|     0| 4838|
# |                        0|     1| 1555|
# +-------------------------+------+-----+

num_cols, cat_cols = get_col_types(df)

##### Label Encoding #####
def get_dtype(col_name):
    return [col[1] for col in df.dtypes if col[0] == col_name][0]

string_cols = []
for col in cat_cols:
    if get_dtype(col) == "string":
        string_cols.append(col)

for col in string_cols:
    indexer = StringIndexer(inputCol=col, outputCol=col + "_label")
    temp_df = indexer.fit(df).transform(df)
    df = temp_df.withColumn(col + "_label", temp_df[col + "_label"].cast("integer"))
    df = df.drop(col)

num_cols, cat_cols = get_col_types(df)

##### One Hot Encoding #####
cat_cols = [col for col in cat_cols if col not in ["exited"]]
outputCols = [col + "_ohe" for col in cat_cols]
encoder = OneHotEncoder(inputCols=cat_cols, outputCols=outputCols)
df = encoder.fit(df).transform(df)

# Target identification
stringIndexer = StringIndexer(inputCol='exited', outputCol='label')
temp_df = stringIndexer.fit(df).transform(df)
df = temp_df.withColumn("label", temp_df["label"].cast("integer"))

# Feature identification
cols = ['creditscore',
        'age',
        'tenure',
        'balance',
        'numofproducts',
        'hascrcard',
        'isactivemember',
        'estimatedsalary',
        'age_that_become_customer',
        'balance_salary_ratio',
        'gender_label',
        'geography_label_ohe',
        'tenure_cat_label_ohe',
        'hascrcard_ohe',
        'gender_label_ohe',
        'numofproducts_ohe',
        'isactivemember_ohe',
        'hascrcard_or_activemember_ohe',
        'age_gender_cat_label_ohe',
        'age_cat_label_ohe']

# Vectorizing
va = VectorAssembler(inputCols=cols, outputCol="features")
va_df = va.transform(df)

# Final dataframe
final_df = va_df.select("features", "label")
final_df.show(5)
# +--------------------+-----+
# |            features|label|
# +--------------------+-----+
# |(31,[0,1,2,4,5,6,...|    1|
# |(31,[0,1,2,3,4,6,...|    0|
# |(31,[0,1,2,3,4,5,...|    1|
# |(31,[0,1,2,4,7,8,...|    0|
# |(31,[0,1,2,3,4,5,...|    0|
# +--------------------+-----+

#####################################################
# GRADIENT BOOSTED TREE CLASSIFIER MODEL
#####################################################

# train, test split
train_df, test_df = final_df.randomSplit([0.7, 0.3], seed=17)

gbm = GBTClassifier(maxIter=100, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df)
y_pred = gbm_model.transform(test_df)

# accuracy
y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()
# 0.8459521468371026

#####################################################
# MODEL TUNING
#####################################################
evaluator = BinaryClassificationEvaluator()

gbm_params = (ParamGridBuilder()
              .addGrid(gbm.maxDepth, [3, 4, 5])
              .addGrid(gbm.maxBins, [30, 40])
              .addGrid(gbm.maxIter, [20, 30])
              .build())

cv = CrossValidator(estimator=gbm,
                    estimatorParamMaps=gbm_params,
                    evaluator=evaluator,
                    numFolds=5)

cv_model = cv.fit(train_df)

# Selected params
best_model = cv_model.bestModel
best_model._java_obj.getMaxDepth()
best_model._java_obj.getMaxBins()
best_model._java_obj.getMaxIter()

# Making predictions
y_pred = cv_model.transform(test_df)

#####################################################
# RESULTS
#####################################################

evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

acc = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "accuracy"})
precision = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "precisionByLabel"})
recall = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "recallByLabel"})
f1 = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "f1"})
roc_auc = evaluator.evaluate(y_pred)

print("accuracy: %f, precision: %f, recall: %f, f1: %f" % (acc, precision, recall, f1))
# accuracy: 0.852835, precision: 0.868963, recall: 0.959091, f1: 0.838195

# Shut down the SparkContext.
sc.stop()