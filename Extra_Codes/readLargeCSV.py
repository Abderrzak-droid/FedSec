import pandas as pd

# Adjust chunksize based on your system's memory (e.g., 100,000 rows per chunk)
CHUNKSIZE = 100000
INPUT_FILE = 'E:\\PFE2025\\Dataset\\Dataset_2\\Phase2\\phase2_NetworkData.csv'
OUTPUT_FILE = 'E:\\PFE2025\\Dataset\\Dataset_2\\Phase2\\MaliciousFlows.csv'

# Check if 'label' column exists in the input file
try:
    columns = pd.read_csv(INPUT_FILE, nrows=0).columns
    if 'label' not in columns:
        raise ValueError("CSV file does not contain a 'label' column.")
except FileNotFoundError:
    raise FileNotFoundError(f"The file {INPUT_FILE} was not found.")

# Process the CSV in chunks to reduce memory usage
header_written = False
for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNKSIZE, dtype={'label': 'int8'}):
    # Filter rows where label is 1
    filtered_chunk = chunk[chunk['label'] == 1]
    
    # Skip writing if no rows matched the filter
    if filtered_chunk.empty:
        continue
    
    # Write to output (append mode)
    filtered_chunk.to_csv(
        OUTPUT_FILE,
        mode='a',          # Append mode
        header=not header_written,  # Write header only once
        index=False        # Do not include row indices
    )
    
    # Update header flag after first write
    if not header_written:
        header_written = True

print("Processing complete. Rows with label=1 saved to", OUTPUT_FILE)