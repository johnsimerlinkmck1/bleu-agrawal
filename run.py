from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')
nltk.download('punkt')
# Example sentences
reference = [['this', 'is', 'a', 'test']]
candidate = ['this', 'is', 'a', 'test2']

# Calculate BLEU score
bleu = sentence_bleu(reference, candidate)
print(f'Bleu Score: {bleu}')

# Calculate ROUGE score
rouge = Rouge()
scores = rouge.get_scores(' '.join(candidate), ' '.join(reference[0]))
print(f'Rouge Score: {scores}')

# Calculate METEOR score
meteor = meteor_score([reference[0]], candidate)
print(f'Meteor Score: {meteor}')