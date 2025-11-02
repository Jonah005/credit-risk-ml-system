from preprocessing import preprocess_data
from model import build_ann
from evaluation import evaluate_model
import subprocess
import sys


# Preprocess data
X_train, X_test, y_train, y_test = preprocess_data('german.data-numeric')

# Convert labels: 2 -> 0
y_train = (y_train == 2).astype(int)
y_test = (y_test == 2).astype(int)

# Build and train model
model = build_ann(X_train.shape[1])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate
evaluate_model(model, X_test, y_test)

# Run model comparison script
print("\n🔁 Running comparison.py...")
subprocess.run([sys.executable, "comparison.py"])