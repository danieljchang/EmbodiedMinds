def calculate_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total

def calculate_precision(predictions, labels):
    true_positive = ((predictions == 1) & (labels == 1)).sum().item()
    false_positive = ((predictions == 1) & (labels == 0)).sum().item()
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

def calculate_recall(predictions, labels):
    true_positive = ((predictions == 1) & (labels == 1)).sum().item()
    false_negative = ((predictions == 0) & (labels == 1)).sum().item()
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

def calculate_f1_score(predictions, labels):
    precision = calculate_precision(predictions, labels)
    recall = calculate_recall(predictions, labels)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def calculate_metrics(predictions, labels):
    accuracy = calculate_accuracy(predictions, labels)
    precision = calculate_precision(predictions, labels)
    recall = calculate_recall(predictions, labels)
    f1_score = calculate_f1_score(predictions, labels)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }