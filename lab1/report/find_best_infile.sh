#!/bin/bash

# Define the input file
input_file="avg_scale_errors.txt"

# Initialize the minimum average and best line variables
min_average=""
best_line=""

# Loop through each line in the input file
while read line; do
    # Extract the numbers from the line
    numbers=$(echo $line | awk '{print $3}')

    # Compute the average of the numbers
    average=$(echo $numbers | awk '{sum=0; for(i=1; i<=NF; i++) sum+=$i; print sum/NF}')

    # Check if this is the first line or if the current average is the new minimum
    if [ -z "$min_average" ] || (( $(echo "$average < $min_average" | bc -l) )); then
        # Update the minimum average and best line variables
        min_average=$average
        best_line=$line
    fi

    # Append "BEST" to the end of the line
    echo "$line BEST" >> "${input_file}.tmp"

done < "$input_file"

# Replace the original file with the modified lines
mv "${input_file}.tmp" "$input_file"

