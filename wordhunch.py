# coding: utf-8

import web
from web import form
import redis
import time
import math
import nltk
from nltk.corpus import brown
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import *

r = redis.Redis(host='localhost', port=6379, db=0)
rr = redis.Redis(host='localhost', port=6379, db=1)

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

def compute_concordance():
	rr.flushdb()
	words = brown.words()
	for w_i in xrange(0, len(words)/1):
		for nearbyword in words[w_i-window:w_i+window+1]:
			rr.zincrby(words[w_i].lower(), nearbyword.lower(), 1)

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
	# match tagged inputs words to correct tagged synonyms
	# take average of current word vector and corpus word vector for similarity calculation
	# normalize word vector; save (hash?) as word_sense
	# do not show syns with sim < median_sim
	all_syns = []
	nostopwords = []
	input_text = input_text.decode('utf-8')
	query = [word.lower() for wordlist in [word_tokenize(sent) for sent in sent_tokenize(input_text)] for word in wordlist]
	for w in query:
		if unicode(w) not in stopwords.words() and unicode(w) not in u".,\'’!?":
			nostopwords.append(w)
	
	for q_i in xrange(0, len(nostopwords)):
		#this might be a bad idea.
		q_word_vect = {}
		for nearbyword in nostopwords[q_i-window:q_i+window+1]:
			nearbyword = nearbyword.lower()
			if nearbyword not in q_word_vect.keys():
				q_word_vect[nearbyword] = 1
			else:
				q_word_vect[nearbyword] += 1
			rr.zincrby(nostopwords[q_i].lower(), nearbyword, 1)
		
		q_word = nostopwords[q_i]

		#this needs to be q_word+':'+q_word_sense+':syns'
		if True: #not rr.exists(q_word+':syns'):
			synonyms = [lemma.name for lemmalist in [synset.lemmas for synset in wordnet.synsets(q_word)] for lemma in lemmalist]
			for s_word in synonyms:
				#q_word_vect = rr.hgetall(q_word)
				s_word_vect = rr.zrangebyscore(s_word, max='+inf', min='0', withscores=True)
				sim_score = sim(q_word_vect, s_word_vect)
				if sim_score:
					rr.zadd(q_word+':syns', s_word, sim_score)
		all_syns.append([q_word, rr.zrevrange(q_word+':syns', start=0, num=5, withscores=True)])
		#change this to zincrby
		rr.hincrby('request_frequency', q_word, 1)
	return all_syns

def get_process_form():
	f = process_form()
	return render.process(f)

def get_recent_results():
	queries = r.sort('recent_queries', start=0, num=10, desc=True)
	texts = r.sort('recent_queries', start=0, num=10, get='*', desc=True)
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
		r.lpush('recent_queries', key)
		r.set(key, user_data.input_text)
		
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
	compute_concordance()
	app.run()
