import sys
from bs4 import BeautifulSoup

blacklist = ['__...__ - сӑмаха каҫӑ евӗр тӑвасси.',
'__aaa|...__ - сӑмахӑн каҫине тепӗр сӑмахпа хатӗрлесси («...» вырӑнне «ааа» пулӗ).',
'__http://chuvash.org|...__ - сӑмах ҫине тулаш каҫӑ лартасси.',
'**...** - хулӑм шрифтпа палӑртасси.',
'~~...~~ - тайлӑк шрифтпа палӑртасси.',
'___...___ - аялтан чӗрнӗ йӗрпе палӑртасси.']


txt = open(sys.argv[1]).read()
newtxt = ''
for line in txt.split('\n'):
#	if line.count('news_tags')>0: continue
	newtxt += line + '\n'
soup = BeautifulSoup(newtxt, 'html.parser')

#soup.find('div',id='hipar_text').find_all('p')
pars = soup.find_all('p')

out = ''
for p in pars:
	class_ = '_'
	style_ = '_'
	if 'class' in p.attrs: class_ = p.attrs['class']
	if 'style' in p.attrs: style_ = p.attrs['style']
	if class_ == '_':
		t = p.get_text().strip()
		if t in blacklist: continue
		if p.find_all('img'): continue
		out += t + '\n'

print(out.strip())
