from preprocessing import preprocess_data
from model import build_ann

# Get preprocessed data
X_train, X_test, y_train, y_test = preprocess_data('german.data-numeric')

# Adjust labels (since labels are 1 and 2, convert 2 -> 0)
y_train = (y_train == 2).astype(int)
y_test = (y_test == 2).astype(int)

# Build ANN
model = build_ann(X_train.shape[1])

# Train ANN
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save model for comparison
model.save("ann_model.h5")
print("✅ ANN model saved as ann_model.h5")
