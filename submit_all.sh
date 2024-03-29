#!/bin/bash

# Save the current directory
original_dir=$(pwd)

# Check if a directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# Verify if the provided argument is a directory
if [ ! -d "$1" ]; then
  echo "Error: $1 is not a directory."
  exit 1
fi

# Navigate to the directory
cd "$1"

# Find all files ending with .sh and submit them using sbatch
find . -type f -name "*.sh" -exec sbatch {} \;

echo "All .sh files in $1 have been submitted."

# Return to the original directory
cd "$original_dir"
