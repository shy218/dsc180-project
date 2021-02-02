
def unigram_encode(data, n_unigrams):
	word_count = {}
	stopwords = nltk.corpus.stopwords.words('english')

	for form in data['full_text']:
	    cleaned_form = re.sub(r'\W',' ', form)
	    cleaned_form = re.sub(r'\s+',' ', cleaned_form)
	    cleaned_form = cleaned_form.lower()
	    tokens = nltk.word_tokenize(cleaned_form)
	    for token in tokens:
	        if token in stopwords:
	            continue
	        if token not in word_count.keys():                 
	            word_count[token] = 1
	        else: 
	            word_count[token] += 1

	most_freq = heapq.nlargest(n_unigrams, word_count, key=word_count.get)

	form_vectors = []
	for form in data['full_text']:
	    cleaned_form = re.sub(r'\W',' ', form)
	    cleaned_form = re.sub(r'\s+',' ', cleaned_form)
	    cleaned_form = cleaned_form.lower()
	    tokens = nltk.word_tokenize(cleaned_form)
	    temp = []
	    for token in most_freq:
	        if token in cleaned_form:                 
	            temp.append(1)
	        else: 
	            temp.append(0)
	    form_vectors.append(temp)

	data['unigram_vec'] = form_vectors

	return

def phrase_encode(data, phrases, treshhold):

	quality_phrases = pd.read_csv(phrases, sep = '\t', header = None)
	
	def clean(text):
    return text.lower()

    quality_phrases['cleaned'] = quality_phrases[1].apply(clean)

	top_phrases = quality_phrases['cleaned'].loc[quality_phrases[0] > treshhold].copy()

    phrase_vectors = []
	for form in data['full_text']:
	    cleaned_form = cleaned_form.lower()
	    temp = []
	    for phrase in top_phrases:
	        if phrase in cleaned_form:                 
	            temp.append(1)
	        else: 
	            temp.append(0)
	    phrase_vectors.append(temp)

	data['phrase_vec'] = phrase_vectors




