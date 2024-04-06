class ConfusionMatrix():
    def __init__(self):
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.true_negative = 0

    def update(self, actual_change, predicted_change):
        """Updates the confusion matrix based on the actual and predicted changes."""
        if actual_change > 0 and predicted_change > 0:
            self.true_positive += 1
        elif actual_change < 0 and predicted_change > 0:
            self.false_positive += 1
        elif actual_change > 0 and predicted_change < 0:
            self.false_negative += 1
        elif actual_change < 0 and predicted_change < 0:
            self.true_negative += 1