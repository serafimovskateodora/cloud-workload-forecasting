def add_safety_margin(predicted_cpu, margin=5.0):
    return min(100.0, predicted_cpu + margin)
