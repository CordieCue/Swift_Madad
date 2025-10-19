import random

# Body parts for more natural phrasing
body_parts = [
    "head", "forehead", "face", "jaw", "scalp", "neck",
    "chest", "stomach", "abdomen", "back", "torso",
    "left arm", "right arm", "arm", "wrist", "hand", "fingers",
    "left leg", "right leg", "leg", "thigh", "knee", "foot", "ankle", "toes"
]

# Common symptoms or conditions
injury_descriptions = [
    "is bleeding", "feels broken", "hurts bad", "got a deep cut", "got a gash",
    "is crushed", "is missing", "is stuck", "won’t move", "is bruised", "is numb", "feels swollen"
]

# Sentence fragments
head_lines = [
    "My {part} {injury}.", "I’m bleeding from my {part}.", "My {part}'s hurt.", "My {part}'s fine.",
    "There’s blood on my {part}.", "My {part}'s cut open."
]

torso_lines = [
    "My chest {injury}.", "My torso's {injury}.", "My back {injury}.", "My stomach's {injury}.",
    "My chest hurts when I breathe.", "My torso's okay.", "Can’t tell about my torso, it’s pinned."
]

limb_lines = [
    "My {part} {injury}.", "I can’t move my {part}.", "My {part}'s stuck under debris.",
    "My {part}'s gone after the blast.", "My {part} feels broken.", "My {part}'s bleeding badly.",
    "My {part}'s fine.", "I can’t feel my {part}."
]

overall_lines = [
    "Everything else feels fine.", "I think I’m okay elsewhere.", "Arms and legs seem alright.",
    "I’m not sure about the rest.", "I feel dizzy.", "I’m freaking out a bit.",
    "Just shaken up.", "Pain’s getting worse.", "I’m scared."
]

# Generate a single first-person trauma report
def generate_natural_trauma_sentence():
    sentences = []

    # Head or face
    if random.random() < 0.7:
        part = random.choice(["head", "scalp", "face", "forehead"])
        injury = random.choice(injury_descriptions)
        line = random.choice(head_lines).format(part=part, injury=injury)
        sentences.append(line)

    # Torso
    if random.random() < 0.6:
        line = random.choice(torso_lines)
        if "{injury}" in line:
            line = line.format(injury=random.choice(injury_descriptions))
        sentences.append(line)

    # Limbs
    limb_part1 = random.choice(["left arm", "right arm", "left leg", "right leg", "hand", "foot"])
    limb_part2 = random.choice(["left leg", "right leg", "arm", "leg", "fingers", "ankle"])
    injury1 = random.choice(injury_descriptions)
    injury2 = random.choice(injury_descriptions)
    line1 = random.choice(limb_lines).format(part=limb_part1, injury=injury1)
    line2 = random.choice(limb_lines).format(part=limb_part2, injury=injury2)
    sentences.append(line1)
    if random.random() < 0.7:
        sentences.append(line2)

    # Overall condition
    if random.random() < 0.6:
        sentences.append(random.choice(overall_lines))

    return " ".join(sentences)

# Generate dataset
def generate_natural_trauma_dataset(n=9000):
    return [f"\"{generate_natural_trauma_sentence()}\",\n" for _ in range(n)]

# Save to file
if __name__ == "__main__":
    data = generate_natural_trauma_dataset(9000)
    with open("natural_style_trauma_sentences.txt", "w") as f:
        f.writelines(data)

    print("✅ Saved 1000 natural-style trauma sentences to 'natural_style_trauma_sentences.txt'")
