def log(message):
    print(f"[LOG] {message}")

def save_model(model, filename):
    import joblib
    joblib.dump(model, filename)