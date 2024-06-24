def transform(data, model):
    features = [f"x{i}" for i in range(5)]
    data = data[features]
    return data.fillna(-9999)
