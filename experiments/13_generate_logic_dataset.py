import json
import random

OUTPUT_FILE = "data/logic_dataset.json"

# Templates designed to induce hallucination if reasoning fails
TEMPLATES = [
    {
        "type": "negation_trap",
        "template": "All {category} are {adjective}. This object is not {adjective}. Therefore, this object is definitely not a"
    },
    {
        "type": "transitive_chain",
        "template": "If {A} implies {B}, and {B} implies {C}, and {C} implies {D}. {A} is true. However, {D} is false. This creates a contradiction because"
    },
    {
        "type": "nested_clauses",
        "template": "The {noun1} that the {noun2} that the {noun3} {verb3} {verb2} {verb1}."
    }
]

# Vocabulary for generation
VOCAB = {
    "category": [
        "wugs", "daxes", "blips", "feps", "kikis", "boubas", "zorks", "glorps", "snarks", "quexes",
        "thabo", "crags", "plinks", "warps", "vorks", "mips", "neeps", "glims", "bops", "tarns"
    ],
    "adjective": [
        "cute", "blue", "fast", "heavy", "loud", "soft", "bright", "sharp", "smooth", "rough",
        "red", "green", "slow", "light", "quiet", "hard", "dark", "dull", "bumpy", "sticky"
    ],
    "noun": ["cat", "dog", "mouse", "rat", "bird", "lion", "tiger", "bear", "wolf", "fox"],
    "verb": ["chased", "bit", "scared", "hugged", "saw", "heard", "kicked", "found", "loved", "liked"],
    "variables": ["P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
}

def generate_dataset():
    dataset = []
    
    # 1. Negation Traps (N=150)
    seen_negations = set()
    while len(dataset) < 150:
        cat = random.choice(VOCAB["category"])
        adj = random.choice(VOCAB["adjective"])
        # Use singular for the final closure
        # Removing 's' or 'es' heuristically for the target check
        cat_singular = cat[:-2] if cat.endswith("es") else cat[:-1]
        
        prompt = f"All {cat} are {adj}. This object is not {adj}. Therefore, this object is definitely not a"
        
        if prompt not in seen_negations:
            dataset.append({
                "id": f"neg_{len(dataset)}",
                "type": "negation_trap",
                "prompt": prompt,
                "target_completion": cat_singular
            })
            seen_negations.add(prompt)

    # 2. Transitive Chains (N=150)
    seen_chains = set()
    while len(dataset) < 300:
        # Pick 4 random variables
        vars = random.sample(VOCAB["variables"], 4)
        A, B, C, D = vars
        prompt = f"If {A} implies {B}, and {B} implies {C}, and {C} implies {D}. {A} is true. However, {D} is false. This creates a contradiction because"
        
        if prompt not in seen_chains:
            dataset.append({
                "id": f"chain_{len(dataset)}",
                "type": "transitive_chain",
                "prompt": prompt,
                "target_completion": "contradiction" # Heuristic check target
            })
            seen_chains.add(prompt)

    # 3. Nested Clauses (N=60)
    seen_nested = set()
    while len(dataset) < 200:
        n1, n2, n3 = random.sample(VOCAB["noun"], 3)
        v1, v2, v3 = random.sample(VOCAB["verb"], 3)
        
        prompt = f"The {n1} that the {n2} that the {n3} {v3} {v2} {v1}."
        
        if prompt not in seen_nested:
            dataset.append({
                "id": f"nested_{len(dataset)}",
                "type": "nested_clauses",
                "prompt": prompt,
                "target_completion": "grammatically_correct" # Hard to check auto, logic is structure
            })
            seen_nested.add(prompt)
            
    return dataset

def main():
    data = generate_dataset()
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Generated {len(data)} samples in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
