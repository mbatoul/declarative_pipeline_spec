from sklearn.model_selection import cross_validate


def evaluate(model, X, y, cv, scores=[]):
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scores,
    )

    output_result = ""
    for score in scores:
        fmt_score = score.lstrip("_", " ").capitalize()
        score_result = cv_results[f"test_{score}"]
        score_result = -score_result if "neg" in score else score_result
        output_result += (
            f"{fmt_score}: {score_result.mean():.3f} +/- {score_result.std():.3f}\n"
        )

    print(output_result)
