
from pathlib import Path
import joblib


class ArabicFAQIntentClassifier:
    """
    Production wrapper for the trained Arabic FAQ question-type classifier.

    Model:
        TF-IDF (char n-grams 3-5) + LinearSVC

    The complete sklearn Pipeline is loaded from disk.
    """

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = (
                Path("/content/arabic_faq_data/models")
                / "intent_classifier.joblib"
            )

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Intent classifier not found: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)

        if not hasattr(self.model, "predict"):
            raise TypeError(
                "Loaded intent model does not implement predict()."
            )

    def predict(self, query):
        """
        Predict the question type for a single Arabic query.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty.")

        prediction = self.model.predict([query])

        return str(prediction[0])

    def predict_batch(self, queries):
        """
        Predict question types for multiple queries.
        """
        if not isinstance(queries, (list, tuple)):
            raise TypeError("queries must be a list or tuple.")

        cleaned = []

        for query in queries:
            if not isinstance(query, str):
                raise TypeError("Every query must be a string.")

            query = query.strip()

            if not query:
                raise ValueError("Queries cannot contain empty strings.")

            cleaned.append(query)

        predictions = self.model.predict(cleaned)

        return [str(x) for x in predictions]

    @property
    def classes(self):
        """
        Return the classifier's known question-type classes.
        """
        if hasattr(self.model, "classes_"):
            return [str(x) for x in self.model.classes_]

        # sklearn Pipeline stores classifier classes inside the final step
        if hasattr(self.model, "named_steps"):
            classifier = self.model.named_steps.get("classifier")

            if classifier is not None and hasattr(classifier, "classes_"):
                return [str(x) for x in classifier.classes_]

        return []


if __name__ == "__main__":
    # Lightweight standalone sanity test
    classifier = ArabicFAQIntentClassifier()

    test_query = "أريد إلغاء اشتراكي نهائياً"

    prediction = classifier.predict(test_query)

    print("Intent classifier loaded successfully.")
    print(f"Test query: {test_query}")
    print(f"Predicted intent: {prediction}")
    print(f"Known classes: {len(classifier.classes)}")
