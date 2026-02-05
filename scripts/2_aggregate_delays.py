from pyspark.sql import SparkSession, functions as F

# Initialize Spark (using smaller resources for the aggregation step)
spark = (SparkSession.builder
         .appName("4_visualize_delays").config("spark.driver.memory", "20g")
         .config("spark.driver.cores", "4")
         .config("spark.executor.memory", "20g")
         .config("spark.executor.cores", "4")
         .config("spark.executor.instances", "10")
         .getOrCreate())

df = spark.read.parquet("data_project/delays_with_coords_and_temps.parquet")

print(f"Columns in the combined dataframe: {df.columns}")
print(f"Total number of rows: {df.count()}")

station_stats = (df
    .filter(F.col("Latitude").isNotNull() & F.col("Longitude").isNotNull())
    .groupBy("station_name", "Latitude", "Longitude")
    .agg(
        F.avg("delay_in_min").alias("avg_delay"),
        F.avg(F.when(F.col("is_canceled") == True, 1).otherwise(0)).alias("cancellation_rate"),
        F.count("*").alias("sample_size")
    )
    # filter out stations with very few observations for a cleaner map
    .filter(F.col("sample_size") > 5)
)

# save to parquet for local visualization
station_stats.write.mode("overwrite").parquet("data_project/aggregated_data/station_delay_stats.parquet")

station_stats_ICE = (df
    .filter(F.col("Latitude").isNotNull() & F.col("Longitude").isNotNull())
    .filter(F.col("train_type") == "ICE")
    .groupBy("station_name", "Latitude", "Longitude")
    .agg(
        F.avg("delay_in_min").alias("avg_delay"),
        F.avg(F.when(F.col("is_canceled") == True, 1).otherwise(0)).alias("cancellation_rate"),
        F.count("*").alias("sample_size")
    )
    .filter(F.col("sample_size") > 5)
)

station_stats_ICE.write.mode("overwrite").parquet("data_project/aggregated_data/station_delay_ICE.parquet")


temperature_delay = (df
    .withColumn("temp_bin", F.round("temperature", 0))
    .groupBy("temp_bin")
    .agg(
        F.avg("delay_in_min").alias("avg_delay"),
        F.avg(F.when(F.col("is_canceled") == True, 1).otherwise(0)).alias("cancellation_rate"),
        F.count("*").alias("sample_size")
    )
    .filter(F.col("sample_size") > 5)
    .orderBy("temp_bin")
)
temperature_delay.write.mode("overwrite").parquet("data_project/aggregated_data/temperature_delay.parquet")

time_of_day_delay = (df
    .withColumn("hour_of_day", F.hour("time"))
    .groupBy("hour_of_day")
    .agg(
        F.avg("delay_in_min").alias("avg_delay"),
        F.avg(F.when(F.col("is_canceled") == True, 1).otherwise(0)).alias("cancellation_rate"),
        F.count("*").alias("sample_size")
    )
    .filter(F.col("sample_size") > 5)
    .orderBy("hour_of_day")
)
time_of_day_delay.write.mode("overwrite").parquet("data_project/aggregated_data/time_of_day_delay.parquet")

train_type_delay = (df
    .groupBy("train_type")
    .agg(
        F.avg("delay_in_min").alias("avg_delay"),
        F.avg(F.when(F.col("is_canceled") == True, 1).otherwise(0)).alias("cancellation_rate"),
        F.count("*").alias("sample_size")
    )
    .filter(F.col("sample_size") > 5)
    .orderBy("train_type")
)
train_type_delay.write.mode("overwrite").parquet("data_project/aggregated_data/train_type_delay.parquet")

# aggregate temperature difference to previous day
from pyspark.sql.window import Window
window_spec = Window.partitionBy().orderBy("time")
temp_diff_delay = (df
    .withColumn("prev_temp", F.lag("temperature").over(window_spec))
    .withColumn("temp_diff", F.col("temperature") - F.col("prev_temp"))
    .withColumn("temp_diff_bin", F.round("temp_diff", 0))
    .groupBy("temp_diff_bin")
    .agg(
        F.avg("delay_in_min").alias("avg_delay"),
        F.avg(F.when(F.col("is_canceled") == True, 1).otherwise(0)).alias("cancellation_rate"),
        F.count("*").alias("sample_size")
    )
    .filter(F.col("sample_size") > 5)
    .orderBy("temp_diff_bin")
)
temp_diff_delay.write.mode("overwrite").parquet("data_project/aggregated_data/temp_diff_delay.parquet")

# day of week delay
day_of_week_delay = (df
    .withColumn("day_of_week", F.date_format("time", "E"))
    .groupBy("day_of_week")
    .agg(
        F.avg("delay_in_min").alias("avg_delay"),
        F.avg(F.when(F.col("is_canceled") == True, 1).otherwise(0)).alias("cancellation_rate"),
        F.count("*").alias("sample_size")
    )
    .filter(F.col("sample_size") > 5)
    .orderBy("day_of_week")
)
day_of_week_delay.write.mode("overwrite").parquet("data_project/aggregated_data/day_of_week_delay.parquet")


traffic_density_delays = (df
    .withColumn("num_trains", F.count("*").over(Window.partitionBy(F.date_trunc("hour", F.col("time")))))
    .withColumn("train_count_bin", F.pow(10, F.round(F.log10(F.col("num_trains")) * 20) / 20))
    .groupBy("train_count_bin")
    .agg(
        F.avg("delay_in_min").alias("avg_delay"),
        F.avg(F.when(F.col("is_canceled") == True, 1).otherwise(0)).alias("cancellation_rate"),
        F.count("*").alias("sample_size")
    )
    .orderBy("train_count_bin")
)

traffic_density_delays.write.mode("overwrite").parquet("data_project/aggregated_data/traffic_density_delays.parquet")

# temperature delay grouped by train type
temperature_delay_by_train_type = (df
    .withColumn("temp_bin", F.round("temperature", 0))
    .groupBy("temp_bin", "train_type")
    .agg(
        F.avg("delay_in_min").alias("avg_delay"),
        F.avg(F.when(F.col("is_canceled") == True, 1).otherwise(0)).alias("cancellation_rate"),
        F.count("*").alias("sample_size")
    )
    .filter(F.col("sample_size") > 5)
    .orderBy("temp_bin", "train_type")
)
temperature_delay_by_train_type.write.mode("overwrite").parquet("data_project/aggregated_data/temperature_delay_by_train_type.parquet")

# aggregate temperature difference to 3 days before
from pyspark.sql.window import Window
window_spec_3day = Window.partitionBy().orderBy("time")
temp_diff_3day_delay = (df
    .withColumn("prev_temp_3day", F.lag("temperature", 3).over(window_spec_3day))
    .withColumn("temp_diff_3day", F.col("temperature") - F.col("prev_temp_3day"))
    .withColumn("temp_diff_3day_bin", F.round("temp_diff_3day", 0))
    .groupBy("temp_diff_3day_bin")
    .agg(
        F.avg("delay_in_min").alias("avg_delay"),
        F.avg(F.when(F.col("is_canceled") == True, 1).otherwise(0)).alias("cancellation_rate"),
        F.count("*").alias("sample_size")
    )
    .filter(F.col("sample_size") > 5)
    .orderBy("temp_diff_3day_bin")
)
temp_diff_3day_delay.write.mode("overwrite").parquet("data_project/aggregated_data/temp_diff_3day_delay.parquet")
