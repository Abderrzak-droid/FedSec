import pandas as pd
from datetime import timedelta
from tqdm import tqdm

def shift_timestamps_and_project_labels(file1_path, file2_path, output_path, trigger_time="12:00:00 AM"):
    # Read both CSV files
    print("Reading CSV files...")
    df1 = pd.read_csv(file1_path)  # Original file with Stage values
    df2 = pd.read_csv(file2_path)  # File that needs timestamp shifting
    
    # Create a copy of the original Timestamp
    df2['Original_Timestamp'] = df2['Timestamp']
    
    # Process timestamps that need shifting
    print("Processing timestamps...")
    times = df2['Timestamp'].str.extract(r'(\d{2}:\d{2}:\d{2} [AP]M)')[0]
    rows_to_shift = times == trigger_time
    indices_to_shift = df2.index[rows_to_shift]
    
    print(f"Found {len(indices_to_shift)} rows to shift")
    
    # Shift timestamps for identified rows
    with tqdm(total=len(indices_to_shift), desc="Shifting timestamps") as pbar:
        for idx in indices_to_shift:
            # Convert to datetime, shift, and convert back to string
            current_time = pd.to_datetime(df2.at[idx, 'Timestamp'], format='%d/%m/%Y %I:%M:%S %p')
            new_time = current_time - pd.Timedelta(hours=8)
            df2.at[idx, 'Timestamp'] = new_time.strftime('%d/%m/%Y %I:%M:%S %p')
            pbar.update(1)
    
    # Convert timestamps to datetime for both dataframes
    print("\nPreparing for label projection...")
    df1['Timestamp'] = pd.to_datetime(df1['Timestamp'], format='%d/%m/%Y %I:%M:%S %p', errors='coerce')
    df2['Timestamp'] = pd.to_datetime(df2['Timestamp'], format='%d/%m/%Y %I:%M:%S %p', errors='coerce')
    
    # Sort both DataFrames by Timestamp
    df1 = df1.sort_values('Timestamp').reset_index(drop=True)
    df2 = df2.sort_values('Timestamp').reset_index(drop=True)
    
    # Shift df1 timestamps by 8 hours
    df1['Timestamp_shifted'] = df1['Timestamp'] + timedelta(hours=8)
    df1['Timestamp_shifted'] = df1['Timestamp_shifted'].dt.strftime('%d/%m/%Y %I:%M:%S %p')
    
    # Define matching criteria
    match_cols = ['Src IP', 'Src Port', 'Dst IP', 'Dst Port']
    
    # Merge dataframes using string timestamps
    print("Projecting labels...")
    merged_df = pd.merge(
        df2,
        df1[['Timestamp_shifted'] + match_cols + ['Stage']],
        left_on=['Original_Timestamp'] + match_cols,
        right_on=['Timestamp_shifted'] + match_cols,
        how='left'
    )
    
    # Update Label column
    merged_df['Label'] = merged_df['Stage']
    
    # Convert Timestamp back to string format if it's not already
    if pd.api.types.is_datetime64_any_dtype(merged_df['Timestamp']):
        merged_df['Timestamp'] = merged_df['Timestamp'].dt.strftime('%d/%m/%Y %I:%M:%S %p')
    
    # Clean up temporary columns
    merged_df = merged_df.drop(columns=['Timestamp_shifted', 'Stage', 'Original_Timestamp'])
    
    # Save the result
    print(f"Saving results to {output_path}")
    merged_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Total rows processed: {len(df2)}")
    print(f"Rows with timestamps shifted: {len(indices_to_shift)}")
    print(f"Rows with labels projected: {merged_df['Label'].notna().sum()}")
    
    return merged_df

# File paths
file1_path = "E:\\PFE2025\\Dataset\\Dataset_1\\CSV Original\\enp0s3-tcpdump-pvt-friday.pcap_Flow.csv"
file2_path = "E:\\FlowsExtractedByUs\\enp0s3-tcpdump-pvt-friday.pcap_Flow.csv"
output_path = "E:\\CSVExtractedTrue\\enp0s3-tcpdump-pvt-friday.pcap_Flow.csv"

# Execute the processing
result_df = shift_timestamps_and_project_labels(file1_path, file2_path, output_path)