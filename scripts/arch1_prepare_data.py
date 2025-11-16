import os, re, json, random, urllib.request

random.seed(42)
os.makedirs("data", exist_ok=True)

N_TRAIN = 10_000
N_EVAL  = 500

def get_sentences():
    raw = ""
    # Önce Tiny Shakespeare'i dene
    try:
        raw = urllib.request.urlopen(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            timeout=20
        ).read().decode("utf-8")
    except Exception:
        pass
    
    # Fallback: zengin metin seti (her zaman ekle) - daha çeşitli ve gerçekçi
    templates = [
        "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid.",
        "Istanbul is the largest city in Turkey. Ankara is the capital of Turkey. Izmir is a major port city in Turkey.",
        "FAISS is a library for similarity search. It uses approximate nearest neighbor algorithms. HNSW is a graph-based index structure.",
        "DPR is a dual-encoder model for question answering. BERT is a transformer-based language model. GPT is a generative pre-trained transformer.",
        "Python is a high-level programming language. JavaScript is used for web development. Java is an object-oriented language.",
        "Machine learning is a subset of artificial intelligence. Deep learning uses neural networks. Natural language processing deals with text.",
        "The Earth orbits around the Sun. The Moon orbits around the Earth. Mars is the fourth planet from the Sun.",
        "Water boils at 100 degrees Celsius. Ice melts at 0 degrees Celsius. The speed of light is approximately 300000 kilometers per second.",
        "Shakespeare wrote Romeo and Juliet. Dickens wrote A Tale of Two Cities. Tolkien wrote The Lord of the Rings.",
        "The Amazon is the longest river in South America. The Nile is the longest river in Africa. The Mississippi is a major river in North America.",
        "Mount Everest is the highest mountain in the world. K2 is the second highest. Kilimanjaro is the highest mountain in Africa.",
        "The Pacific Ocean is the largest ocean. The Atlantic Ocean is the second largest. The Indian Ocean is the third largest.",
        "Einstein developed the theory of relativity. Newton discovered the laws of motion. Darwin proposed the theory of evolution.",
        "The Renaissance was a period of cultural rebirth in Europe. The Industrial Revolution changed manufacturing. The Information Age began with computers.",
        "DNA contains genetic information. RNA helps in protein synthesis. Proteins are essential for life functions.",
        "London is the capital of England. Tokyo is the capital of Japan. Moscow is the capital of Russia.",
        "The Great Wall of China is the longest wall in the world. The Eiffel Tower is located in Paris. The Statue of Liberty is in New York.",
        "Socrates was a Greek philosopher. Plato was a student of Socrates. Aristotle was a student of Plato.",
        "The United States has 50 states. California is the most populous state. Alaska is the largest state by area.",
        "The human heart has four chambers. The brain controls the nervous system. The lungs are responsible for breathing.",
    ]
    raw += " " + " ".join(templates * 2500)  # Daha fazla çeşitlilik için artırıldı
    
    sents = re.split(r'(?<=[.!?])\s+', raw)
    sents = [re.sub(r"\s+", " ", s).strip() for s in sents if len(s) > 40]
    sents = [s.rstrip(".") + "." for s in sents]
    random.shuffle(sents)
    return sents

def make_train_passages(sents):
    # Daha çeşitli passage'lar için sıralamayı karıştır ve tekrarları azalt
    train = sents[:N_TRAIN]
    # Tekrarları azaltmak için unique passage'ları tercih et
    seen = set()
    unique_train = []
    for t in train:
        t_norm = " ".join(t.lower().split())
        if t_norm not in seen and len(t) > 30:  # Minimum uzunluk
            seen.add(t_norm)
            unique_train.append(t)
        if len(unique_train) >= N_TRAIN:
            break
    # Eğer yeterli unique passage yoksa, kalanları ekle
    while len(unique_train) < N_TRAIN and len(train) > len(unique_train):
        for t in train:
            if t not in unique_train:
                unique_train.append(t)
                if len(unique_train) >= N_TRAIN:
                    break
    
    with open("data/passages_train.jsonl", "w", encoding="utf-8") as f:
        for i, t in enumerate(unique_train[:N_TRAIN]):
            f.write(json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n")
    print(f"OK: data/passages_train.jsonl ({len(unique_train[:N_TRAIN])})")

def make_eval_qa(sents):
    eval_pairs = []
    # Pattern 1: "X is Y"
    for s in sents:
        m = re.search(r"([A-Z][^.!?]{3,40}) is ([A-Z][^.!?]{1,50})\.", s)
        if not m:
            m = re.search(r"([A-Z][^.!?]{3,40}) is ([^.!?]{1,50})\.", s)
        if m:
            subj = m.group(1).strip()
            obj  = m.group(2).strip()
            q = f"What is {subj}?"
            a = obj.split(",")[0].split(" and ")[0].strip()
            eval_pairs.append({"question": q, "answers": [a]})
        if len(eval_pairs) >= N_EVAL:
            break
    
    # Pattern 2: "X wrote Y" -> "Who wrote Y?"
    if len(eval_pairs) < N_EVAL:
        for s in sents:
            m = re.search(r"([A-Z][a-z]+) wrote ([A-Z][^.!?]{5,50})\.", s)
            if m:
                author = m.group(1).strip()
                work = m.group(2).strip()
                q = f"Who wrote {work}?"
                eval_pairs.append({"question": q, "answers": [author]})
            if len(eval_pairs) >= N_EVAL:
                break
    
    # Pattern 3: "X is the Y in Z" -> "What is the Y in Z?"
    if len(eval_pairs) < N_EVAL:
        for s in sents:
            m = re.search(r"([A-Z][^.!?]{3,40}) is the ([^.!?]{3,30}) in ([A-Z][^.!?]{3,30})\.", s)
            if m:
                entity = m.group(1).strip()
                desc = m.group(2).strip()
                location = m.group(3).strip()
                q = f"What is the {desc} in {location}?"
                eval_pairs.append({"question": q, "answers": [entity]})
            if len(eval_pairs) >= N_EVAL:
                break
    
    # Fallback: Basit QA çiftleri
    if len(eval_pairs) < N_EVAL:
        fallback_qa = [
            {"question": "What is the capital of France?", "answers": ["Paris"]},
            {"question": "Which city is largest in Turkey?", "answers": ["Istanbul"]},
            {"question": "What is the capital of Germany?", "answers": ["Berlin"]},
            {"question": "What is the capital of Italy?", "answers": ["Rome"]},
            {"question": "What is the capital of Spain?", "answers": ["Madrid"]},
            {"question": "What is the capital of Turkey?", "answers": ["Ankara"]},
            {"question": "Who wrote Romeo and Juliet?", "answers": ["Shakespeare"]},
            {"question": "What is the longest river in South America?", "answers": ["The Amazon"]},
            {"question": "What is the highest mountain in the world?", "answers": ["Mount Everest"]},
            {"question": "What is the largest ocean?", "answers": ["The Pacific Ocean"]},
        ]
        # Fallback'leri tekrarlayarak N_EVAL'a ulaş
        while len(eval_pairs) < N_EVAL:
            eval_pairs.extend(fallback_qa)
        eval_pairs = eval_pairs[:N_EVAL]
    with open("data/gold_eval.jsonl", "w", encoding="utf-8") as f:
        for o in eval_pairs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"OK: data/gold_eval.jsonl ({len(eval_pairs)})")

if __name__ == "__main__":
    sents = get_sentences()
    make_train_passages(sents)
    make_eval_qa(sents[N_TRAIN:])
