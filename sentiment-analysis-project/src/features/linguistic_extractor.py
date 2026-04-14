import spacy

nlp = spacy.load("en_core_web_sm")

class LinguisticFeatureExtractor:
    def __init__(self):
        pass

    def extract_structure(self,text):
        doc = nlp(text)
        features = []

        for token in doc:
            # 1. Dependency Tagging: How is this word related to its 'head'?
            # Example: 'nsubj' (nominal subject), 'amod' (adjectival modifier)
            dep_relation = f"{token.dep_}_{token.head.text}_{token.text}"
            features.append(dep_relation)

            if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
                features.append(f"adj_noun_{token.text}_{token.head.text}")

            return features
        
if __name__ == '__main__':
    extractor = LinguisticFeatureExtractor()
    sample = "The plot was incredibly dull, but the lead actor gave a masterful performance."
    
    struct_features = extractor.extract_structure(sample)
    
    print("\n--- LINGUISTIC STRUCTURE FEATURES ---")
    print(struct_features)
    for f in struct_features:
        if "adj_noun" in f or "nsubj" in f:
            print(f"Found Relation: {f}")