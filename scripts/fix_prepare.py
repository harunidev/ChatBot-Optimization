#!/usr/bin/env python3
import re

file_path = "/Users/harunisik/Desktop/ChatBot-Optimization/scripts/arch1_prepare_data.py"

with open(file_path, 'r') as f:
    content = f.read()

# Fix the broken generate_synthetic_qa function
# Find and replace the problematic section
old_section = r'''    count = 0
    # Try more pairs than needed to find good candidates
    for idx in selected_indices:
        if count >= num_questions:
            break
            
            passage = passages[idx]
        sentences = passage.replace("?", ".").replace("!", ".").split(".")
        
        # Filter for quality sentences
        valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) > 8] 
        
        if not valid_sentences:
            continue
            
        qa_pairs.append({
            "question": random.choice(valid_sentences),
            "answer": random.choice(valid_sentences)
        })
        count += 1
        
    return qa_pairs'''

new_section = '''    count = 0
    # Try more pairs than needed to find good candidates
    for idx in selected_indices:
        if count >= num_questions:
            break
            
        passage = passages[idx]
        sentences = passage.replace("?", ".").replace("!", ".").split(".")
        
        # Filter for quality sentences
        valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) > 8] 
        
        if not valid_sentences:
            continue
            
        # Use sentence as question
        question = random.choice(valid_sentences)
        
        qa_pairs.append({
            "id": str(count),
            "question": question,
            "answers": [question],  # Sentence itself as answer for now
            "gold_passage_id": str(idx),
            "gold_passage_text": passage
        })
        count += 1
        
    return qa_pairs'''

content = content.replace(old_section, new_section)

with open(file_path, 'w') as f:
    f.write(content)

print("Fixed arch1_prepare_data.py")
