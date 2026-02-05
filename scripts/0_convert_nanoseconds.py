import pandas as pd
import glob
import os

file_list = sorted(glob.glob("trains/data-202[45]-*.parquet"))
if not os.path.exists("cleaned_trains"):
    os.mkdir("cleaned_trains")

columns = ["station_name", "delay_in_min", "is_canceled", "train_type", "time"]

for file_path in file_list:
    print(f"Processing {os.path.basename(file_path)}...")
    pdf = pd.read_parquet(file_path, columns=columns)
    # Cast the column explicitly to microsecond resolution in Pandas
    # This changes the data type from datetime64[ns] to datetime64[us]
    for col in pdf.select_dtypes(include=['datetime64[ns]']).columns:
        pdf[col] = pdf[col].astype('datetime64[us]')
    print(f"Columns: {pdf.dtypes}")

    # Save using pyarrow with the specific timestamp_precision flag
    # This forces the Parquet metadata to be INT64 (TIMESTAMP(MICROS,true))
    output_name = "cleaned_" + file_path
    print(f"Saving {output_name}...")
    pdf.to_parquet(
        output_name,
        engine="pyarrow",
        coerce_timestamps="us", # Force microseconds
        allow_truncated_timestamps=True
    )