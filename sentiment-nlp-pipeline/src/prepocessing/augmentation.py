import random
import nltk
from nltk.corpus import wordnet
from deep_translator import GoogleTranslator

class DataAugmentor:
    def __init__(self):
        nltk.download('wordnet', quiet=True)
        
    def synonym_replacement(self, text, n=1):
        """Replaces n words in the sentence with synonyms."""
        words = text.split()
        new_words = words.copy()
        random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
        
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = []
            for syn in wordnet.synsets(random_word):
                for l in syn.lemmas():
                    synonyms.append(l.name())
            
            if len(synonyms) > 1:
                synonym = random.choice(list(set(synonyms)))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
        return ' '.join(new_words)

    def inject_noise(self, text, p=0.1):
        """Randomly deletes characters to simulate typos."""
        if len(text) <= 1: return text
        chars = list(text)
        new_chars = [c for c in chars if random.random() > p]
        return "".join(new_chars)

    def back_translate(self, text, target_lang='de'):
        """Translates to German and back to English to paraphrase."""
        try:
            # English to German
            german_text = GoogleTranslator(source='en', target=target_lang).translate(text)
            print("German Text : ", german_text)
            # German back to English
            english_text = GoogleTranslator(source=target_lang, target='en').translate(german_text)
            return english_text
        except Exception as e:
            return text # Fallback to original if API fails

if __name__ == "__main__":
    augmentor = DataAugmentor()
    sample = "The script was very slow and the acting was mediocre."
    
    print(f"Original: {sample}")
    print(f"Synonym:  {augmentor.synonym_replacement(sample, n=2)}")
    print(f"Noise:    {augmentor.inject_noise(sample)}")
    print(f"Back-Trans: {augmentor.back_translate(sample)}")