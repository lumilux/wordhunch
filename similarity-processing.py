from nltk.corpus import brown
from nltk.corpus import wordnet
from nltk.corpus import stopwords

#preprocess
def compute_concordance():
	words = brown.words()
	window = 3
	for w_i in xrange(0, len(words)/1000):
		for nearbyword in [words[w_i-window:w_i+window+1]]:
			rr.hincrby(words[w_i], nearbyword, 1)

def sim(dict_a, dict_b):
	keys = intersect(dict_a.keys(), dict_b.keys())
	numerator = sum([float(dict_a[key]) * float(dict_b[key]) for key in keys])
	denominator = sqrt(sum([float(dict_a[key])**2 for key in keys])) * sqrt(sum([float(dict_b[key])**2 for key in keys]))
	return numerator / denominator

def get_syns(input_text):
	all_syns = {}
	nostopwords = []
	query = [word.lower() for wordlist in [word_tokenize(sent) for sent in sent_tokenize(input_text)] for word in wordlist]
	for w in query:
		if w not in stopwords.words():
			nostopwords.append(w)
	for q_word in nostopwords:
		synonyms = [lemma.name for lemmalist in [synset.lemmas for synset in wordnet.synsets(q_word)] for lemma in lemmalist]
		for s_word in synonyms:
			q_word_vect = rr.hgetall(q_word)
			s_word_vect = rr.hgetall(s_word)
			sim = sim(q_word, s_word)
			rr.zadd(q_word+':syns', sim, s_word)
		all_syns[q_word] = rr.zrange(q_word+':syns', start=0, end=-1, withscores=True)
	return all_syns