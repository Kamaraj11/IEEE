class ProgressTracker:
    def __init__(self):
        pass

    def evaluate_progress(self, old_prob, new_prob, class_name):
        """
        Evaluate if a condition has improved or worsened based on the predicted probability of the SAME class.
        """
        if old_prob is None or old_prob <= 0:
            return "No previous valid data to compare."
            
        diff = new_prob - old_prob
        
        # If the probability of a malignant or precarious condition INCREASES, it's worsening.
        # Malignant/precarious classes: akiec, bcc, mel
        malignant_classes = ['akiec', 'bcc', 'mel']
        
        if class_name in malignant_classes:
            if diff > 5.0:
                return f"WORSENING: Probability of {class_name.upper()} increased by {diff:.1f}%."
            elif diff < -5.0:
                return f"IMPROVING: Probability of {class_name.upper()} decreased by {abs(diff):.1f}%."
            else:
                return f"STABLE: No significant change ({diff:+.1f}%)."
        else:
            # Benign typically implies tracking the likelihood of benign status. 
            # If Benign probability drops significantly, something else is dominating.
            if diff > 5.0:
                return f"STABLE / BENIGN ASSURANCE: Likelihood increased by {diff:.1f}%."
            elif diff < -5.0:
                return f"ATTENTION REQUIRED: Confidence in benign diagnosis dropped by {abs(diff):.1f}%."
            else:
                return f"STABLE: No significant change ({diff:+.1f}%)."
