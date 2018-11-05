import math, pickle

def viterbi(transition, emission, sequence, state, evaluate=None, debug=False):
	dp = [[(-1,-1)]*len(state) for i in range(len(sequence))]

	# dp[0] 'start' to sequence[0]
	for j in range(len(state)):
		dp[0][j] = (math.log(transition['START'][state[j]]) + math.log(emission[state[j]][sequence[0]]) , -1)

	# print(dp)
	# dp[i+1] sequence[i] to sequence[i+1]
	for i in range(0, len(sequence)-1):
		for j in range(len(state)):
			max_likelihood = -100000000
			bp = 0
			emission_prob = math.log(emission[state[j]][sequence[i+1]])
			for k in range(len(state)):
				transition_prob = math.log(transition[state[k]][state[j]])
				likelihood = dp[i][k][0] + transition_prob + emission_prob

				if likelihood > max_likelihood:
					max_likelihood = likelihood
					bp = k

			dp[i+1][j] = (max_likelihood, bp)

	bp = 0
	max_likelihood = -100000000
	for k in range(0, len(state)):
		likelihood = dp[-1][k][0] + math.log(transition[state[k]]['END'])
		if debug: print(likelihood)
		if likelihood > max_likelihood:
			max_likelihood = likelihood
			bp = k

	tags = []
	for i in reversed(range(0, len(sequence))):
		tags.append(state[bp])
		bp = dp[i][bp][1]
	tags.reverse()

	if debug:
		print('| START', end='\t|')
		for i in range(len(sequence)): print(sequence[i], end='\t|')
		print("END\t|")
		for i in range(len(sequence)): print('|-----', end='\t')
		print("|")
		for j in range(len(states)):
			print("|", states[j], end='\t')
			for i in range(len(sequence)):
				bp = state[dp[i][j][1]] if i > 0 else 'START'
				print("|{:.4f},{}".format(dp[i][j][0], bp), end='\t')
			print("|")
			for i in range(len(sequence)): print('|-----', end='\t')
			print("|")
		print("|Prediction", end='\t|')
		for i in range(len(sequence)):
			print("{}".format(tags[i]), end='\t|')
		print("\n----------------------------")

	if not evaluate: return tags
	else: evaluate(tags, dp)

if __name__ == '__main__':

	'''
	transition = {
		'START' : {'N': math.exp(-1), 'V': math.exp(-2), 'END': 0.0},
		'N' : {'N': math.exp(-3), 'V': math.exp(-1), 'END': math.exp(-1)},
		'V' : {'N': math.exp(-1), 'V': math.exp(-3), 'END': math.exp(-1)},
	}
	emission = {
		'N': {'they': math.exp(-2), 'can': math.exp(-3), 'fish': math.exp(-3)},
		'V': {'they': math.exp(-10), 'can': math.exp(-1), 'fish': math.exp(-3)},
	}

	sequence = ['they', 'can', 'fish']
	states = ['N', 'V'] 
	'''
	transition = {
		'START' : {'H': 0.6, 'L': 0.4, 'END': 0.0},
		'H' : {'H': 0.4, 'L': 0.4, 'END': 0.2},
		'L' : {'H': 0.2, 'L': 0.5, 'END': 0.3}
	}
	emission = {
		'H': {'A': 0.2, 'C':0.3, 'G':0.3, 'T':0.2},
		'L': {'A': 0.3, 'C':0.2, 'G':0.2, 'T':0.4}
	}

	sequence = ['G', 'C', 'A', 'C', 'T', 'G']
	states = ['H', 'L'] 

	viterbi(transition, emission, sequence, states, debug=True)








