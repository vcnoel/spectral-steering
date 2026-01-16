import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )

    candidates = [
        # 1. The original Wug (Varied)
        "All wugs are cute. This animal is not cute. Therefore, this animal is definitely not a",
        
        # 2. Counterfactual + Negation
        "If the sky were green, then grass would be blue. The sky is indeed green. Therefore, grass is",
        
        # 3. Double Negation Trap
        "It is not true that the impossible is not possible. Therefore, the impossible is",
        
        # 4. Self-Reference
        "This sentence is false. Therefore, the previous sentence is",
        
        # 5. Nested Logic (Depth 3)
        "If P implies Q, and Q implies R, and R implies S. P is true. However, S is false. This creates a contradiction because",
        
        # 6. Center Embedding (Memory stress)
        "The rat the cat the dog chased killed ate the malt. Who ate the malt?",
        
        # 7. Inverse Logic
        "Only glips are glops. This object is a glop. Therefore, it must be a",
        
        # 8. False Premise Trap
        "Since the moon is made of cheese, and all cheese is refrigerated, the moon must be",
    ]

    print("\n--- Searching for Hallucinations ---")
    for i, prompt in enumerate(candidates):
        print(f"\n[{i+1}] Prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Output: {response[len(prompt):].strip()}")

if __name__ == "__main__":
    main()
