
class IntentAwareReranker:
    """
    Intent-aware reranker for Arabic FAQ retrieval.

    Final score:
        final_score = E5_score + alpha * intent_match

    Where:
        intent_match = 1 if candidate intent matches predicted query intent
        intent_match = 0 otherwise

    The default alpha (0.03) comes from the observed diagnostic experiment.
    It should be treated as a deployment candidate, not as a formally
    validation-tuned hyperparameter.
    """

    def __init__(self, alpha=0.03):
        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be numeric.")

        if alpha < 0:
            raise ValueError("alpha must be >= 0.")

        self.alpha = float(alpha)

    def rerank(self, candidates, predicted_intent, top_k=None):
        """
        Rerank retrieved candidates using predicted question type.

        Parameters
        ----------
        candidates : list[dict]
            Retrieval candidates returned by ArabicFAQRetriever.search().

        predicted_intent : str
            Intent predicted by ArabicFAQIntentClassifier.

        top_k : int or None
            Number of candidates to return.
            If None, return all candidates.

        Returns
        -------
        list[dict]
            Reranked candidates with:
                - original E5 score
                - intent_match
                - final_score
                - rerank_rank
        """

        if not isinstance(candidates, list):
            raise TypeError("candidates must be a list.")

        if not isinstance(predicted_intent, str):
            raise TypeError("predicted_intent must be a string.")

        predicted_intent = predicted_intent.strip()

        if not predicted_intent:
            raise ValueError("predicted_intent cannot be empty.")

        if not candidates:
            return []

        reranked = []

        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise TypeError("Each candidate must be a dictionary.")

            if "score" not in candidate:
                raise KeyError("Candidate is missing 'score'.")

            if "question_type" not in candidate:
                raise KeyError(
                    "Candidate is missing 'question_type'. "
                    "Production metadata must include question_type."
                )

            e5_score = float(candidate["score"])

            candidate_intent = str(
                candidate["question_type"]
            ).strip()

            intent_match = int(
                candidate_intent == predicted_intent
            )

            final_score = (
                e5_score +
                self.alpha * intent_match
            )

            item = dict(candidate)

            item["e5_score"] = e5_score
            item["candidate_intent"] = candidate_intent
            item["intent_match"] = intent_match
            item["final_score"] = final_score

            reranked.append(item)

        # Stable deterministic sorting:
        # 1. Higher final score
        # 2. Higher original E5 score
        # 3. Lower original rank
        reranked.sort(
            key=lambda x: (
                x["final_score"],
                x["e5_score"],
                -int(x.get("rank", 10**9))
            ),
            reverse=True
        )

        for new_rank, item in enumerate(reranked, 1):
            item["rerank_rank"] = new_rank

        if top_k is not None:
            if not isinstance(top_k, int):
                raise TypeError("top_k must be an integer or None.")

            if top_k <= 0:
                raise ValueError("top_k must be > 0.")

            reranked = reranked[:top_k]

        return reranked


if __name__ == "__main__":
    # Lightweight standalone sanity test
    reranker = IntentAwareReranker(alpha=0.03)

    candidates = [
        {
            "rank": 1,
            "score": 0.86,
            "question_type": "pricing or fees"
        },
        {
            "rank": 2,
            "score": 0.87,
            "question_type": "policy (refund, cancellation, warranty, exchange)"
        }
    ]

    predicted_intent = "pricing or fees"

    results = reranker.rerank(
        candidates,
        predicted_intent,
        top_k=2
    )

    print("Reranker loaded successfully.")

    for result in results:
        print(
            result["rerank_rank"],
            result["candidate_intent"],
            result["e5_score"],
            result["final_score"]
        )
