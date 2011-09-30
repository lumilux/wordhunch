# coding: utf-8

import web
from web import form
import redis
import time
import math
import nltk
from nltk.corpus import brown
from nltk.corpus import treebank
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import *

r_msg = redis.Redis(host='localhost', port=6379, db=0)
r = redis.Redis(host='localhost', port=6379, db=1)

render = web.template.render('templates/', base='base')

urls = (
	'/help', 'help',
	'/index', 'index',
	'/words', 'words',
	'/', 'index'
)

app = web.application(urls, globals())

validtext = form.regexp(r"^.+$", 'Please enter some text.')

process_form = form.Form(
	form.Textbox("input_text", validtext, description=""),
	form.Button("Submit", type="submit")
)

# 3 might be the best
# greater than 5 takes forever.
window = 5
punctuation = '(),-.:?!;&|$%@#~^–—\/\"\'<>[]{}+=_'
tag_dict = {'N':'n', 'V':'v', 'ADJ':'a', 'ADV':'r'}
pos_list = 'nvar'
all_sents = []

def simp_pos(tag):
	if len(tag) > 0:
		tag = nltk.tag.simplify.simplify_tag(tag).lower()
	if tag in pos_list:
		return tag
	if tag not in tag_dict:
		return None
	else:
		return tag_dict[tag]
		
def simp_tag(tag):
	pos = simp_pos(tag)
	if not pos:
		return '?'	#other
	else:
		return pos

def tagkey(pair):
	word, tag = pair
	#print(pair)
	if (isinstance(word, unicode) or isinstance(word, str)) and isinstance(tag, str):
		return simp_tag(tag)+':'+word

def compute_concordance(words):
	for w_i in xrange(0, len(words)):
		for nearpair in words[w_i-window:w_i+window+1]:
			nearword, neartag = nearpair
			word, tag = words[w_i]
			w_pair_key = tagkey((word.lower(),tag))
			nearpair_key = tagkey((nearword.lower(),neartag))
			#print(w_pair_key+' '+nearpair_key)
			r.zincrby(w_pair_key, nearpair_key, 1)

def preprocess_corpora():
	brown_words = brown.tagged_words(simplify_tags=True)
	treebank_words = treebank.tagged_words(simplify_tags=True)
	'''
	#this takes forever.
	bwog_corpus = nltk.corpus.PlaintextCorpusReader('../bwog-corpus-txt', '.*\.txt')
	bwog_sents = bwog_corpus.sents(bwog_corpus.fileids())
	bwog_words = []
	for s_i in xrange(0, len(bwog_sents)/100000):
		#TODO: skip punctuation
		simp_tagged_sent = [(word,simp_tag(tag)) for word,tag in nltk.pos_tag(bwog_sents[s_i])]
		bwog_words.extend(simp_tagged_sent)
	'''
	all_tagged_words = brown_words + treebank_words #+ bwog_words
	all_sents = brown.sents() + treebank.sents() #+ bwog_sents
	compute_concordance(all_tagged_words)

# this takes about half an hour.
def cache_syns():
	for sent in all_sents:
		print("cached "+sent)
		get_syns(sent)

def get_syn_lemmas(word, pos):
	ret = [lemma for lemmalist in [synset.lemmas for synset in wordnet.synsets(word, pos)] for lemma in lemmalist]
	if not ret:
		ret = [lemma for lemmalist in [synset.lemmas for synset in wordnet.synsets(word, None)] for lemma in lemmalist]
	#print(word+' '+str(pos))
	return ret

def sim(dict_a, pairs_b):
	dict_b = dict(pairs_b)
	keys = set(dict_a.keys()).intersection(set(dict_b.keys()))
	numerator = sum([float(dict_a[key]) * float(dict_b[key]) for key in keys])
	den1 = math.sqrt(sum([float(dict_a[key])**2 for key in keys]))
	den2 = math.sqrt(sum([float(dict_b[key])**2 for key in keys]))
	
	if len(keys) == 0 or den1 == 0 or den2 == 0:
		return 0.
	return numerator / (den1 * den2)

def get_syns(input_text):
	#TODO
	#+use sorted set instead of hash for word vectors
	#+only use sorted set members with score > 1
	#+adjust context window
	# lemmatize corpus words
	# lemmatize input words
	# remove number words
	# remove helping verbs
	#+tag input words (install NumPy)
	# use Jython and CRFTagger to tag Bwog corpus and input
	# match tagged inputs words to correct tagged synonyms
	# take average of current word vector and corpus word vector for similarity calculation
	# normalize word vector; save (hash?) as word_sense
	# do not show syns with sim < median_sim
	# automate creation of :syns sorted sets
	all_syns = []
	nostopwords = []
	input_text = input_text.decode('utf-8')
	query = [word.lower() for wordlist in [word_tokenize(sent) for sent in sent_tokenize(input_text)] for word in wordlist]
	tagged_query = nltk.pos_tag(query)
	for pair in tagged_query:
		word, tag = pair
		if unicode(word) not in stopwords.words() and tag not in punctuation:
			nostopwords.append((word,tag))
	
	for q_i in xrange(0, len(nostopwords)):
		#this might be a bad idea.
		q_pair_vect = {}
		for nearword,neartag in nostopwords[q_i-window:q_i+window+1]:
			nearpair_key = tagkey((nearword.lower(), neartag))
			if nearpair_key not in q_pair_vect:
				q_pair_vect[nearpair_key] = 1
			else:
				q_pair_vect[nearpair_key] += 1
			word, tag = nostopwords[q_i]
			r.zincrby(tagkey((word.lower(), tag)), nearpair_key, 1)
		
		q_pair = nostopwords[q_i]
		q_word, q_tag = q_pair
		q_pair_key = tagkey(q_pair)
		q_pair_syns_key = q_pair_key+':syns'
		q_pos = simp_pos(q_tag)
		#print(q_pair_syns_key)

		if not r.exists(q_pair_syns_key):
			syn_lemmas = get_syn_lemmas(q_word, q_pos)
			#print(syn_lemmas)
			#syn_lemmas = [syn.lemmas for syn in wordnet.synsets(q_word, q_pos)]
			for lemma in syn_lemmas:
				syn_word = lemma.name
				syn_pos = lemma.synset.pos #what should we do with s: words? (adj. satellites)
				syn_pair_key = tagkey((syn_word, syn_pos))
				if syn_word == q_word:
					r.zrem(q_pair_syns_key, syn_pair_key)
					continue
				#q_pair_vect = r.hgetall(q_pair)
				s_pair_vect = r.zrangebyscore(syn_pair_key, max='+inf', min='0', withscores=True)
				sim_score = sim(q_pair_vect, s_pair_vect)
				r.zadd(q_pair_syns_key, syn_pair_key, sim_score)
		q_syns = [syn[2:] for syn,score in r.zrevrange(q_pair_syns_key, start=0, num=4, withscores=True)]
		print(q_word+': '+str(q_syns))
		all_syns.append([q_word, q_syns])
		#change this to zincrby
		r.hincrby('request_frequency', q_pair_key, 1)
	return all_syns

def get_process_form():
	f = process_form()
	return render.process(f, base=None)

def get_recent_results():
	queries = r_msg.sort('recent_queries', start=0, num=10, desc=True)
	texts = r_msg.sort('recent_queries', start=0, num=10, get='*', desc=True)
	qtdict = zip(queries, texts)
	return render.results(qtdict)

class help:
	def GET(self):
		return render.help()

class index:
	def GET(self):
		return get_recent_results()

class words:
	def GET(self):
		f = process_form()
		return render.process(f)
	
	def POST(self):
		f = process_form()
		if not f.validates():
			return self.GET()
		
		text = web.input().input_text
		syn_dict = get_syns(text)
		
		key = (int)(time.time()*100)
		r_msg.lpush('recent_queries', key)
		r_msg.set(key, text)
		
		return render.syns(text, syn_dict)
		

if __name__ == "__main__":
	#r_msg.flushdb()
	r.flushdb()
	preprocess_corpora()
	#cache_syns()
	app.run()
