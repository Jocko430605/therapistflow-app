def generate_recommendations(user_id):
    suggestions = [
        "Take 15 minutes of quiet reflection.",
        "Move one non-critical session to tomorrow.",
        "Review recent therapy notes for positive moments.",
        "Log mood and burnout markers.",
        "Connect briefly with a peer or colleague."
    ]

    if user_id == "therapist_103":
        suggestions.append("Add hourly micro-breaks — you’ve been pushing hard lately.")

    return suggestions
