import random

with open('with_bug/enhanced_complex_sentence.txt', mode='r', encoding='utf-8') as file:
    complex_sentences = file.readlines()

with open('with_bug/enhanced_simple_sentence.txt', mode='r', encoding='utf-8') as file:
    simple_sentences = file.readlines()

random_nums = random.sample(list(range(len(complex_sentences))), len(complex_sentences))

new_complex_sentences = []
new_simple_sentences = []
for num in random_nums:
    new_complex_sentences.append(complex_sentences[num])
    new_simple_sentences.append(simple_sentences[num])

train_complex = new_complex_sentences[:8000]
train_simple = new_simple_sentences[:8000]
test_complex = new_complex_sentences[8000:]
test_simple = new_simple_sentences[8000:]

with open('temp/train_complex.txt', mode='a', encoding='utf-8') as file:
    for s in train_complex:
        file.write(s)

with open('temp/train_simple.txt', mode='a', encoding='utf-8') as file:
    for s in train_simple:
        file.write(s)

with open('temp/test_complex.txt', mode='a', encoding='utf-8') as file:
    for s in test_complex:
        file.write(s)

with open('temp/test_simple.txt', mode='a', encoding='utf-8') as file:
    for s in test_simple:
        file.write(s)

