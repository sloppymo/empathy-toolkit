"""
Generate sample data with empathy scores for visualization demo
"""
import pandas as pd
import numpy as np
import os

# Create sample data
data = {
    "response_id": range(1, 11),
    "context": [
        "Patient: I'm feeling anxious about this procedure.",
        "Client: I'm not sure I can afford this treatment.",
        "Friend: I just got some bad news about my health.",
        "Student: I failed my exam and I'm devastated.",
        "Colleague: My project proposal was rejected.",
        "Customer: Your product doesn't work as advertised.",
        "Patient: The side effects are really bothering me.",
        "Client: I don't understand these financial terms.",
        "Friend: I'm going through a difficult breakup.",
        "Student: I'm struggling to keep up with the workload."
    ],
    "response": [
        "This is a routine procedure with minimal risk.",
        "I understand your concern about the cost. Let's discuss some options that might make this more affordable for you.",
        "That sounds tough. Let me know if you want to talk about it.",
        "You need to study harder next time. These things happen.",
        "I understand how disappointing that must be. Would you like to talk about how to improve it for next time?",
        "Let me check into that for you. Can you tell me more about what's happening?",
        "Those side effects are completely normal. Everyone has them.",
        "I understand these terms can be confusing. Let me explain them in simpler language.",
        "There are plenty of fish in the sea. You'll feel better soon.",
        "I hear you're feeling overwhelmed. Let's look at ways to manage your workload better."
    ],
    "empathy_cognitive": [3.0, 7.0, 4.0, 2.0, 8.0, 6.0, 2.0, 8.0, 3.0, 7.0],
    "empathy_emotional": [2.0, 5.0, 5.0, 1.0, 7.0, 4.0, 1.0, 6.0, 3.0, 8.0],
    "empathy_behavioral": [2.0, 8.0, 6.0, 1.0, 7.0, 7.0, 1.0, 7.0, 2.0, 7.0],
    "empathy_contextual": [2.0, 6.0, 3.0, 1.0, 6.0, 5.0, 2.0, 7.0, 2.0, 6.0],
    "empathy_cultural": [0.0, 4.0, 2.0, 0.0, 4.0, 3.0, 0.0, 5.0, 1.0, 4.0]
}

# Add total empathy score (average of all dimensions)
data["empathy_total"] = [
    round(sum([c, e, b, ct, cu])/5, 1) 
    for c, e, b, ct, cu in zip(
        data["empathy_cognitive"], 
        data["empathy_emotional"], 
        data["empathy_behavioral"], 
        data["empathy_contextual"],
        data["empathy_cultural"]
    )
]

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_file = "sample_enhanced_dataset.csv"
df.to_csv(output_file, index=False)

print(f"Sample dataset created: {output_file}")
