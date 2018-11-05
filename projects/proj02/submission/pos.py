import pickle

def save_pickle(data, filename='sample.pickle'):
    with open(filename, mode='wb') as f:
        pickle.dump(data, f)

def normalize(word):
    w = word.lower()
    if w.isdigit(): return "NUM"
    else: return w
'''
tf = {}
index = 1

def process(filename):
	with open(filename, 'r') as f:
		sentences = f.readlines()
		for stn in sentences:
			for token in stn.rstrip('\n').split(' '):
				tag = token[-1]
				word = normalize(token[0:-2])
				# print word, tag
				if word not in tf: tf[word] = 0
				tf[word] += 1

process('trn.pos')

word2index = {}
index2word = {}
index = 1
index2word[0] = 'UNK'
K = 3
for w, freq in tf.items():
    if w not in word2index:
        if freq < K:
            word2index[w] = 0
        else:
            word2index[w] = index
            index2word[index] = w
            index += 1
V = index #the vocab size including 'unk'
print(V)

save_pickle(word2index, 'word2index.pickle')
save_pickle(index2word, 'index2word.pickle')

def loadFeature(filename):
	features = []
	with open(filename, 'r') as f:
		sentences = f.readlines()
		for stn in sentences:
			feature = []
			for token in stn.rstrip('\n').split(' '):
				tag = token[-1]
				word = normalize(token[0:-2])
				# print(word)
				idx = 0
				if word in word2index: idx = word2index[word]
				feature.append((idx, tag))
				# if idx not in index2word: print(word, idx, tag)
			features.append(feature)
	return features

trnset = loadFeature('trn.pos')
devset = loadFeature('dev.pos')

save_pickle(trnset, 'trnset.pickle')
save_pickle(devset, 'devset.pickle')
'''


word2index = pickle.load(open('word2index.pickle', 'rb'))
index2word = pickle.load(open('index2word.pickle', 'rb'))
V = len(index2word)
trnset = pickle.load(open('trnset.pickle', 'rb'))
devset = pickle.load(open('devset.pickle', 'rb'))

# print(word2index)
# print(index2word)

states = ['START', 'A', 'C', 'D', 'M', 'N', 'O', 'P', 'R', 'V', 'W', 'END']
N = len(states)

'''
transition_cnt = {s: {ss:0 for ss in states} for s in states}
emission_cnt = {s: [0 for i in range(len(index2word))] for s in states}

for stn in trnset:
	last_tag = 'START'
	for word, tag in stn:
		transition_cnt[last_tag][tag] += 1
		emission_cnt[tag][word] += 1
		last_tag = tag
	transition_cnt[last_tag]['END'] += 1

save_pickle(transition_cnt, 'transition_cnt.pickle')
save_pickle(emission_cnt, 'emission_cnt.pickle')
'''

transition_cnt = pickle.load(open('transition_cnt.pickle', 'rb'))
emission_cnt = pickle.load(open('emission_cnt.pickle', 'rb'))

# print(transition_cnt)
# print(emission_cnt)

def mle(get_transition_prob, get_emission_prob, ext):
	transition = {}
	emission = {}

	with open('jw7jb-tprob{}.txt'.format(ext), 'w') as f:
		for state, res in transition_cnt.items():
			if state in {'END'}: continue
			transition[state] = {}
			total = 0
			for k, cnt in res.items():
				total += cnt
			for k, cnt in res.items():
				prob = get_transition_prob(cnt, total)
				transition[state][k] = prob
				if k in {'START'}: continue
				f.write('{}, {}, {}\n'.format(state, k, prob))

	with open('jw7jb-eprob{}.txt'.format(ext), 'w') as f:
		for state, res in emission_cnt.items():
			if state in {'START', 'END'}: continue
			emission[state] = {}
			total = sum(res)
			for k in range(len(res)):
				prob = get_emission_prob(res[k], total)
				emission[state][k] = prob
				#handle comma in csv
				f.write('{}, {}, {}\n'.format(state, index2word[k], prob))

	return transition, emission

def div(cnt, total):
	return cnt/total if total != 0 else 0

alpha = 500
beta = 0.05

def transition_smooth(cnt, total):
	return (cnt+alpha)/(total+alpha*N)
def emission_smooth(cnt, total):
	return (cnt+beta)/(total+beta*V)
# transition, emission = mle(div, div, '')
transition, emission = mle(transition_smooth, emission_smooth, '-smoothed')

save_pickle(transition, 'transition.pickle')
save_pickle(emission, 'emission.pickle')
'''
transition = pickle.load(open('transition.pickle', 'rb'))
emission = pickle.load(open('emission.pickle', 'rb'))
# '''
# print(transition)

from viterbi import viterbi
state = ['A', 'C', 'D', 'M', 'N', 'O', 'P', 'R', 'V', 'W']

correct = 0
total = 0
def get_acc(label):
	def compare(pred, dp):
		global correct
		assert(len(pred) == len(label))
		# print(pred, label)
		correct += sum([1 for i in range(len(pred)) if pred[i] == label[i] ])
	return compare

for stn in devset:
	words = []
	tags = []
	for i in range(len(stn)):
		words.append(stn[i][0]) 
		tags.append(stn[i][1])

	total += len(tags)
	viterbi(transition, emission, words, state, evaluate=get_acc(tags))

acc = correct / total
print(alpha, beta, acc)
# '''
with open('tst.word', 'r') as f, open('jw7jb-viterbi.txt', 'w') as results:
	sentences = f.readlines()
	for stn in sentences:
		words = []
		tokens = []
		for token in stn.rstrip('\n').split(' '):
			word = normalize(token)
			# print(word)
			idx = 0
			if word in word2index: idx = word2index[word]
			words.append(idx)
			tokens.append(token)

		preds = viterbi(transition, emission, words, state)

		results.write(" ".join(["%s\\%s"%(token, tag) for token, tag in zip(tokens, preds)]))
		results.write("\n")


