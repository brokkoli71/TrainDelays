## generates a dataframe with train delays, coordinates and weather data
import os
import sys

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType

# Ensure the worker uses the python inside your unpacked environment
os.environ['PYSPARK_PYTHON'] = "./environment/bin/python"
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = (SparkSession.builder
         .appName("build_delay_dataframe")
         .config("yarn.nodemanager.resource.detect-hardware-capabilities", "True")
         .config("yarn.nodemanager.resource.memory-mb", "196608")
         .config("spark.driver.memory", "20g")
         .config("spark.driver.cores", "4")
         .config("spark.executor.memory", "20g")
         .config("spark.executor.cores", "4")
         .config("spark.executor.instances", "20")
         .config("spark.dynamicAllocation.enabled", "false")
         # Ensure Arrow is optimized for the Pandas conversion
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         # to use conda environment packaged as a spark archive
         .config("spark.archives", "hdfs:///user/fe25rav/envs/environment.tar.gz#environment")
         .getOrCreate())

path = "data_project/cleaned_trains/data-202[45]-*.parquet"

print(f"Reading files from {path}...")
trains = spark.read.parquet(path)

print(f"Columns in the combined dataframe: {trains.columns}")
print(f"Total number of rows: {trains.count()}")

coords = (spark.read.format("csv")
    .option("header", "true")
    .option("sep", ";")
    .option("quote", '"')
    .load("data_project/stations/zHV_aktuell_csv.2026-01-05.csv"))

# filter for stops (S) only and process "," separated floats
coords = (coords
    .filter(F.col("Type").contains("S"))
    .select(
        F.col("Name"),
        F.regexp_replace(F.col("Latitude"), ",", ".").cast("float").alias("Latitude"),
        F.regexp_replace(F.col("Longitude"), ",", ".").cast("float").alias("Longitude")
    ))

# normalize names for joining
def normalize_col(col):
    # 1. Take the first word only
    # 2. Lowercase it
    # 3. Remove non-alphanumeric characters
    first_word = F.split(col, '[ -]').getItem(0)
    lower_word = F.lower(first_word)
    return F.regexp_replace(lower_word, '[^a-z0-9]', '')

trains = trains.withColumn("norm_name", normalize_col(F.col("station_name")))
coords = coords.withColumn("norm_name", normalize_col(F.col("Name")))

# take the median of the coords of each city (median for outlier robustness)
coords_median = (coords.groupBy("norm_name")
              .agg(
                  F.median("Latitude").alias("Latitude"),
                  F.median("Longitude").alias("Longitude")
              ))

unique_stations = trains.select("norm_name").distinct()
used_stations = unique_stations.join(
    F.broadcast(coords_median),
    on="norm_name",
    how="inner"
).select("norm_name", "Latitude", "Longitude")

trains_with_coords = trains.join(F.broadcast(coords_median), on="norm_name", how="left")

missing_count = trains_with_coords.filter(F.col("Latitude").isNull()).select("norm_name").distinct().count()
print(f"Stations with missing coordinates: {missing_count} and unique stations: {used_stations.count()}")
print("Preview of result:")
trains_with_coords.show(5)

# ship weather files to all executors
weather_hdfs_path = "weather/"
spark.sparkContext.addFile(weather_hdfs_path + "tas_hyras_1_2024_v6-1_de.nc")
spark.sparkContext.addFile(weather_hdfs_path + "tas_hyras_1_2025_v6-1_de.nc")
spark.sparkContext.addFile(weather_hdfs_path + "tas_hyras_1_2026_v6-1_de.nc")

def batch_lookup(pdf_iterator):
    import xarray as xr
    import pandas as pd
    from pyproj import Transformer
    from pyspark import SparkFiles

    # transformer is used to transform lon, lat to x,y format of HYRAS
    # Define the transformer: WGS84 (4326) to ETRS89 LAEA (3035)
    # always_xy=True ensures output is (x, y) order
    transformer = Transformer.from_crs("epsg:4326", "epsg:3035", always_xy=True)

    local_files = [
        SparkFiles.get("tas_hyras_1_2024_v6-1_de.nc"),
        SparkFiles.get("tas_hyras_1_2025_v6-1_de.nc"),
        SparkFiles.get("tas_hyras_1_2026_v6-1_de.nc")
    ]

    # .load() puts the ~200MB into the 20GB of executor RAM for instant lookups
    with xr.open_mfdataset(local_files, combine="nested", concat_dim="time", chunks={}) as ds:
        ds_loaded = ds.load()
        temperature = ds_loaded["tas"]

        for pdf in pdf_iterator:
            if pdf.empty:
                yield pdf
                continue

            # Transforms Lon/Lat degrees into Meters (x/y) used by HYRAS
            projected_x, projected_y = transformer.transform(
                pdf['Longitude'].values,
                pdf['Latitude'].values
            )

            times = pd.to_datetime(pdf['time']).dt.normalize()

            # Vectorized xarray lookup
            temps = temperature.sel(
                x=xr.DataArray(projected_x, dims="z"),
                y=xr.DataArray(projected_y, dims="z"),
                time=xr.DataArray(times.values, dims="z"),
                method='nearest'
            ).values

            pdf['temperature'] = temps.astype(float)
            yield pdf

trains_with_coords = trains_with_coords.withColumn("temperature", F.lit(None).cast(DoubleType()))
result_sdf = trains_with_coords.mapInPandas(batch_lookup, schema=trains_with_coords.schema)

result_sdf.write.mode("overwrite").parquet("data_project/delays_with_coords_and_temps.parquet")

result_sdf.show(5)