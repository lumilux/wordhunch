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
window = 4
punctuation_tags = ['(',')',',','--','.',':']
tag_dict = {'N': 'n', 'V': 'v', 'ADJ': 'a', 'ADV': 'r'}

def simp_pos(tag):
	if q_tag not in tag_dict:
		return None
	else:
		return tag_dict[q_tag]
		
def simp_tag(tag):
	pos = simp_pos(tag)
	if not pos:
		return ''
	else:
		return pos

def tagkey(pair):
	word, tag = pair
	if isinstance(word, str) and isinstance(tag, str):
		return simp_tag(tag)+':'+word

def compute_concordance(words):
	for w_i in xrange(0, len(words)):
		for nearword,neartag in words[w_i-window:w_i+window+1]:
			word, tag = words[w_i]
			r.zincrby(tagkey((word.lower(),tag)), tagkey((nearword.lower(),neartag)), 1)

def preprocess_corpora():
	brown_words = brown.tagged_words(simplify_tags=True)
	treebank_words = treebank.tagged_words(simplify_tags=True)
	
	#this takes forever.
	bwog_corpus = nltk.corpus.PlaintextCorpusReader('bwog-corpus-txt', '.*\.txt')
	bwog_sents = bwog_corpus.sents(bwog_corpus.fileids())
	bwog_words = []
	for s_i in xrange(0, len(bwog_sents)/1000):
		bwog_words.extend(nltk.pos_tag(bwog_sents[s_i]))
	
	all_tagged_words = brown_words + treebank_words + bwog_words

def sim(dict_a, pairs_b):
	dict_b = dict(pairs_b)
	keys = set(dict_a.keys()).intersection(set(dict_b.keys()))
	numerator = sum([float(dict_a[key]) * float(dict_b[key]) for key in keys])
	den1 = math.sqrt(sum([float(dict_a[key])**2 for key in keys]))
	den2 = math.sqrt(sum([float(dict_b[key])**2 for key in keys]))
	
	if len(keys) == 0 or den1 == 0 or den2 == 0:
		return None
	return numerator / (den1 * den2)

def get_syns(input_text):
	#TODO
	# use sorted set instead of hash for word vectors
	# only use sorted set members with score > 1
	# adjust context window
	# lemmatize corpus words
	# lemmatize input words
	# remove number words
	# remove helping verbs
	# tag input words (install NumPy)
	# use Jython and CRFTagger to tag Bwog corpus and input
	# match tagged inputs words to correct tagged synonyms
	# take average of current word vector and corpus word vector for similarity calculation
	# normalize word vector; save (hash?) as word_sense
	# do not show syns with sim < median_sim
	all_syns = []
	nostopwords = []
	input_text = input_text.decode('utf-8')
	query = [word.lower() for wordlist in [word_tokenize(sent) for sent in sent_tokenize(input_text)] for word in wordlist]
	tagged_query = nltk.pos_tag(query)
	for word,pos in tagged_query:
		if unicode(word) not in stopwords.words() and pos not in punctuation_tags:
			nostopwords.append((word,pos))
	
	for q_i in xrange(0, len(nostopwords)):
		#this might be a bad idea.
		q_pair_vect = {}
		for nearword,neartag in nostopwords[q_i-window:q_i+window+1]:
			nearpair_key = tagkey(nearword.lower(), neartag)
			if nearpair_key not in q_pair_vect:
				q_pair_vect[nearpair_key] = 1
			else:
				q_pair_vect[nearpair_key] += 1
			word, tag = nostopwords[q_i]
			r.zincrby(tagkey(word.lower(), tag), nearpair_key, 1)
		
		q_pair = nostopwords[q_i]
		q_word, q_tag = q_pair
		q_pair_key = tagkey(q_pair)
		q_pair_syns_key = q_pair_key+':syns'
		q_pos = simp_pos(q_tag)

		if not r.exists(q_pair_syns_key):
			syn_lemmas = [lemma for lemmalist in [synset.lemmas for synset in wordnet.synsets(q_word, q_pos)] for lemma in lemmalist]
			#syn_lemmas = [syn.lemmas for syn in wordnet.synsets(q_word, q_pos)]
			for lemma in syn_lemmas:
				syn_word = lemma.name
				syn_pos = lemma.synset.pos
				syn_pair_key = tagkey(syn_word, syn_pos)
				#q_pair_vect = r.hgetall(q_pair)
				s_pair_vect = r.zrangebyscore(syn_pair_key, max='+inf', min='0', withscores=True)
				sim_score = sim(q_pair_vect, s_pair_vect)
				if sim_score:
					r.zadd(q_pair_syns_key, syn_pair_key, sim_score)
		all_syns.append([q_pair_key, r.zrevrange(q_pair_syns_key, start=0, num=5, withscores=True)])
		#change this to zincrby
		r.hincrby('request_frequency', q_pair_key, 1)
	return all_syns

def get_process_form():
	f = process_form()
	return render.process(f)

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
		process = get_process_form()
		
		results = get_recent_results()
		
		return render.index(process, results)

	def POST(self):
		f = process_form()
		if not f.validates():
			return self.GET()
		
		user_data = web.input()
		key = (int)(time.time()*100) #+"_"+str(web.ctx.ip)
		r_msg.lpush('recent_queries', key)
		r_msg.set(key, user_data.input_text)
		
		raise web.seeother('/')

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
		
		return render.syns(text, syn_dict)
		

if __name__ == "__main__":
	r.flushdb()
	preprocess_corpora()
	app.run()
