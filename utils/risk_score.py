def calculate_risk_score(features):
    """
    Calculates a rule-based imaging risk score
    from extracted imaging biomarkers.

    Returns:
        score (0-100)
        risk_level
        interpretation
    """

    score = 0

    volume = features["Volume"]
    mean = features["Mean Intensity"]
    std = features["Standard Deviation"]
    maximum = features["Maximum Intensity"]
    minimum = features["Minimum Intensity"]

    # -------------------------
    # Volume (0-30)
    # -------------------------

    if volume < 3000:
        score += 10
    elif volume < 6000:
        score += 20
    else:
        score += 30

    # -------------------------
    # Mean Intensity (0-25)
    # -------------------------

    if mean < 0.25:
        score += 10
    elif mean < 0.40:
        score += 20
    else:
        score += 25

    # -------------------------
    # Texture Variation (Std Dev)
    # -------------------------

    if std < 0.02:
        score += 10
    elif std < 0.05:
        score += 20
    else:
        score += 25

    # -------------------------
    # Maximum Intensity
    # -------------------------

    if maximum > 0.80:
        score += 10
    else:
        score += 5

    # -------------------------
    # Minimum Intensity
    # -------------------------

    if minimum < 0.20:
        score += 10
    else:
        score += 5

    # -------------------------
    # Risk Category
    # -------------------------

    if score < 40:
        level = "Low Risk"
        interpretation = (
            "Imaging biomarkers suggest relatively low-risk characteristics."
        )

    elif score < 70:
        level = "Moderate Risk"
        interpretation = (
            "Imaging biomarkers indicate moderate-risk characteristics."
        )

    else:
        level = "High Risk"
        interpretation = (
            "Imaging biomarkers indicate high-risk characteristics."
        )

    return score, level, interpretation