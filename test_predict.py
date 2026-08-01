from src.predict import predict_text

sample_text = """
Artificial Intelligence has transformed modern industries by enabling
automation, improving efficiency, and supporting better decision making.
"""

result = predict_text(sample_text)

print("=" * 50)

print("Prediction :", result["prediction"])
print("Confidence :", result["confidence"])

print("=" * 50)