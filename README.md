# Setup environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip

# Unzip dataset
unzip "UCI HAR Dataset.zip"

# Move dataset
mv "UCI HAR Dataset" data/

# Train models
python3 run_all.py
