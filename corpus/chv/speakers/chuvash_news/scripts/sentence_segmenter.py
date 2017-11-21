import sys, re

intxt = sys.stdin.read()

abbrs = ['чӑв.']
abbr_id = 0

abbr = re.compile('([А-Я]+\.[А-Я]+\.)')
for m in abbr.findall(intxt):
	abbrs.append(m)
abbr = re.compile('([А-Я]+\.)')
for m in abbr.findall(intxt):
	abbrs.append(m)

abbrs = list(set(abbrs))
abbr_to_id = {}
for a in abbrs:
	abbr_to_id[a] = '#' + str(abbr_id) + '#'
	intxt = intxt.replace(a, abbr_to_id[a])
	abbr_id += 1

sent_id = 1

for par in intxt.split('\n'):
	lines = re.sub(r'([\.!\?]+) ', r'\1\n', par).split('\n')
	for line in lines:
		for k,v in abbr_to_id.items():
			line = line.replace(v, k)
		line = line.strip()
		if line == '': continue
		print(str(sent_id).zfill(4) + '\t' +line)
		sent_id += 1
