"""
AWS Glue ETL Job — IMDb Dataset Preprocessing (PySpark)
Reads raw CSV from S3, cleans text, encodes labels, saves processed data.

Arguments:
  --JOB_NAME        : Glue job name (injected automatically)
  --S3_INPUT_PATH   : s3://bucket/data/IMDB Dataset.csv
  --S3_OUTPUT_PATH  : s3://bucket/processed/
"""
import sys
import re
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import (
    col, lower, regexp_replace, length, when, trim, split, size
)

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'S3_INPUT_PATH', 'S3_OUTPUT_PATH'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

print(f"Reading raw data from: {args['S3_INPUT_PATH']}")
df = spark.read \
    .option("header", "true") \
    .option("escape", '"') \
    .option("multiLine", "true") \
    .csv(args['S3_INPUT_PATH'])

print(f"Raw record count: {df.count()}")
df.printSchema()
df.show(3, truncate=80)

# --- ETL Transformations ---

# 1. Remove HTML line breaks
df = df.withColumn('review_clean', regexp_replace(col('review'), r'<br\s*/?>', ' '))

# 2. Remove non-alphabetic characters (keep spaces)
df = df.withColumn('review_clean', regexp_replace(col('review_clean'), r'[^a-zA-Z\s]', ''))

# 3. Lowercase
df = df.withColumn('review_clean', lower(col('review_clean')))

# 4. Trim whitespace
df = df.withColumn('review_clean', trim(regexp_replace(col('review_clean'), r'\s+', ' ')))

# 5. Encode sentiment label: positive=1, negative=0
df = df.withColumn('target', when(col('sentiment') == 'positive', 1).otherwise(0))

# 6. Add word count feature
df = df.withColumn('word_count', size(split(col('review_clean'), ' ')))

# 7. Add original review length
df = df.withColumn('original_length', length(col('review')))

# 8. Remove rows with empty reviews
df = df.filter(col('review_clean').isNotNull() & (length(col('review_clean')) > 10))

print(f"Processed record count: {df.count()}")

# Sentiment distribution
print("Sentiment distribution after ETL:")
df.groupBy('sentiment').count().show()

print(f"Saving processed data to: {args['S3_OUTPUT_PATH']}")
df.select('review', 'review_clean', 'sentiment', 'target', 'word_count', 'original_length') \
  .coalesce(1) \
  .write \
  .mode('overwrite') \
  .option('header', 'true') \
  .csv(args['S3_OUTPUT_PATH'])

job.commit()
print("Glue ETL job completed successfully!")
