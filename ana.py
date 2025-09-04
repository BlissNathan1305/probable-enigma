import random

def create_anagram(word):
    word_list = list(word)
    random.shuffle(word_list)
    return ''.join(word_list)

word = "listen"  # You can replace this with any word
anagram = create_anagram(word)
print(f"Original Word: {word}")
print(f"Anagram: {anagram}")
