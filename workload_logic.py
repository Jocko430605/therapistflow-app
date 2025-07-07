def calculate_workload_score(sessions_today, hours, intensity):
    intensity_levels = {"low": 1, "moderate": 2, "high": 3}
    intensity_score = intensity_levels.get(intensity.lower(), 2)

    score = round((sessions_today * 0.4 + hours * 0.5 + intensity_score * 1.2), 2)
    return score
