from lib2to3.pgen2 import token
import math, re
from queue import Empty


# A simple tokenizer. Applies case folding
def tokenize(s):
    tokens = s.lower().split()
    trimmed_tokens = []
    for t in tokens:
        if re.search('\w', t):
            # t contains at least 1 alphanumeric character
            t = re.sub('^\W*', '', t) # trim leading non-alphanumeric chars
            t = re.sub('\W*$', '', t) # trim trailing non-alphanumeric chars
        trimmed_tokens.append(t)
    return trimmed_tokens

# A most-frequent class baseline
class Baseline:
    def __init__(self, klasses):
        self.train(klasses)

    def train(self, klasses):
        # Count classes to determine which is the most frequent
        klass_freqs = {}
        for k in klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        self.mfc = sorted(klass_freqs, reverse=True, 
                          key=lambda x : klass_freqs[x])[0]
    
    def classify(self, test_instance):
        print (self.mfc)

class Lexicon:
    def __init__(self, klasses):
        self.train(klasses)

    def train(self, klasses):
        # Count classes to determine which is the most frequent
        klass_freqs = {}
        for k in klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        self.mfc = sorted(klass_freqs, reverse=True,key=lambda x : klass_freqs[x])[0]
    
    def classify(self, test_instance):
        pos_count = 0
        neg_count = 0
        tokens = tokenize(test_instance)
        for index in range(len(tokens)):
            if pos_words.count(tokens[index]) != 0:
                pos_count = pos_count + 1
            elif neg_words.count(tokens[index]) != 0:
                neg_count = neg_count + 1
        if pos_count == neg_count:
            print ('neutral')
        elif pos_count > neg_count:
            print ('positive')
        elif neg_count > pos_count:
            print ('negative')

class NaiveBayes:
    neg_dict = {}
    pos_dict = {}
    neut_dict = {}
    comm_dict = {}
    klass_freqs = {}
    totKlasses = 0
    pos_count = 0
    neg_count = 0
    neut_count = 0
    
    def __init__(self, klasses):
        self.train(klasses)
        line1 = 0
        for line2 in train_klasses:
            if line2 == 'positive':
                tokens = tokenize(train_texts[line1])
                for index in range(len(tokens)):
                    self.pos_count = self.pos_count + 1
                    self.pos_dict[tokens[index]] = self.pos_dict.get(tokens[index], 0) + 1
                    self.comm_dict[tokens[index]] = self.comm_dict.get(tokens[index], 0) + 1
                line1 = line1 + 1
            elif ( line2 == 'negative'):
                tokens = tokenize(train_texts[line1])
                for index in range(len(tokens)):
                    self.neg_count = self.neg_count + 1
                    self.neg_dict[tokens[index]] = self.neg_dict.get(tokens[index], 0) + 1  
                    self.comm_dict[tokens[index]] = self.comm_dict.get(tokens[index], 0) + 1
                line1 = line1 + 1
            elif ( line2 == 'neutral'):
                tokens = tokenize(train_texts[line1])
                for index in range(len(tokens)):
                    self.neut_count = self.neut_count + 1
                    self.neut_dict[tokens[index]] = self.neut_dict.get(tokens[index], 0) + 1  
                    self.comm_dict[tokens[index]] = self.comm_dict.get(tokens[index], 0) + 1
                line1 = line1 + 1

    def train(self, klasses):
        # Count classes to determine which is the most frequent
        for k in klasses:
            self.totKlasses = self.totKlasses + 1
            self.klass_freqs[k] = self.klass_freqs.get(k, 0) + 1

        self.mfc = sorted(self.klass_freqs, reverse=True,key=lambda x : self.klass_freqs[x])[0]
    
    def classify(self, test_instance):
        maxPosProb = math.log10(self.klass_freqs.get('positive')/self.totKlasses)
        maxNegProb = math.log10(self.klass_freqs.get('negative')/self.totKlasses)
        maxNeutProb = math.log10(self.klass_freqs.get('neutral')/self.totKlasses)
        maxclass = 'neutral'
        tokens = tokenize(test_instance)
    
        for index in range(len(tokens)):
            if ((self.pos_dict.get(tokens[index])!=None) and (self.neg_dict.get(tokens[index])!=None) and (self.neut_dict.get(tokens[index])!=None)):
                maxPosProb = maxPosProb + math.log10((self.pos_dict.get(tokens[index]) + 1) / (self.pos_count + len(self.comm_dict)))
                maxNegProb = maxNegProb + math.log10((self.neg_dict.get(tokens[index]) + 1) / (self.neg_count + len(self.comm_dict)))
                maxNeutProb = maxNeutProb + math.log10((self.neut_dict.get(tokens[index]) + 1) / (self.neut_count + len(self.comm_dict)))

            elif (self.pos_dict.get(tokens[index])!=None and (self.neg_dict.get(tokens[index])!=None) and (self.neut_dict.get(tokens[index])==None)): 
                maxPosProb = maxPosProb + math.log10((self.pos_dict.get(tokens[index]) + 1) / (self.pos_count + len(self.comm_dict)))
                maxNegProb = maxNegProb + math.log10((self.neg_dict.get(tokens[index]) + 1) / (self.neg_count + len(self.comm_dict)))
                maxNeutProb = maxNeutProb + math.log10((0 + 1) / (self.neut_count + len(self.comm_dict)))

            elif(self.pos_dict.get(tokens[index])!=None and (self.neg_dict.get(tokens[index])==None) and (self.neut_dict.get(tokens[index])!=None)):
                maxPosProb = maxPosProb + math.log10((self.pos_dict.get(tokens[index]) + 1) / (self.pos_count + len(self.comm_dict)))
                maxNegProb = maxNegProb + math.log10((0 + 1) / (self.neg_count + len(self.comm_dict)))
                maxNeutProb = maxNeutProb + math.log10((self.neut_dict.get(tokens[index]) + 1) / (self.neut_count + len(self.comm_dict)))

            elif(self.pos_dict.get(tokens[index])!=None and (self.neg_dict.get(tokens[index])==None) and (self.neut_dict.get(tokens[index])==None)):
                maxPosProb = maxPosProb + math.log10((self.pos_dict.get(tokens[index]) + 1) / (self.pos_count + len(self.comm_dict)))
                maxNegProb = maxNegProb + math.log10((0 + 1) / (self.neg_count + len(self.comm_dict)))
                maxNeutProb = maxNeutProb + math.log10((0 + 1) / (self.neut_count + len(self.comm_dict)))

            elif(self.pos_dict.get(tokens[index])==None and (self.neg_dict.get(tokens[index])!=None) and (self.neut_dict.get(tokens[index])!=None)):
                maxPosProb = maxPosProb + math.log10((0 + 1) / (self.pos_count + len(self.comm_dict)))
                maxNegProb = maxNegProb + math.log10((self.neg_dict.get(tokens[index]) + 1) / (self.neg_count + len(self.comm_dict)))
                maxNeutProb = maxNeutProb + math.log10((self.neut_dict.get(tokens[index]) + 1) / (self.neut_count + len(self.comm_dict)))
            
            elif(self.pos_dict.get(tokens[index])==None and (self.neg_dict.get(tokens[index])!=None) and (self.neut_dict.get(tokens[index])==None)):
                maxPosProb = maxPosProb + math.log10((0 + 1) / (self.pos_count + len(self.comm_dict)))
                maxNegProb = maxNegProb + math.log10((self.neg_dict.get(tokens[index]) + 1) / (self.neg_count + len(self.comm_dict)))
                maxNeutProb = maxNeutProb + math.log10((0 + 1) / (self.neut_count + len(self.comm_dict)))

            elif(self.pos_dict.get(tokens[index])==None and (self.neg_dict.get(tokens[index])==None) and (self.neut_dict.get(tokens[index])!=None)):
                maxPosProb = maxPosProb + math.log10((0 + 1) / (self.pos_count + len(self.comm_dict)))
                maxNegProb = maxNegProb + math.log10((0 + 1) / (self.neg_count + len(self.comm_dict)))
                maxNeutProb = maxNeutProb + math.log10((self.neut_dict.get(tokens[index]) + 1) / (self.neut_count + len(self.comm_dict)))

            elif(self.pos_dict.get(tokens[index])==None and (self.neg_dict.get(tokens[index])==None) and (self.neut_dict.get(tokens[index])==None)):
                maxPosProb = maxPosProb + math.log10((0 + 1) / (self.pos_count + len(self.comm_dict)))
                maxNegProb = maxNegProb + math.log10((0 + 1) / (self.neg_count + len(self.comm_dict)))
                maxNeutProb = maxNeutProb + math.log10((0 + 1) / (self.neut_count + len(self.comm_dict)))
        
        if (maxNeutProb > maxNegProb and maxNeutProb > maxPosProb):
            maxclass = 'neutral'
        elif (maxNegProb > maxPosProb and maxNegProb > maxNeutProb):
            maxclass = 'negative'
        elif (maxPosProb > maxNegProb and maxPosProb > maxNeutProb):
            maxclass = 'positive'
        elif (maxPosProb == maxNegProb and maxPosProb > maxNeutProb):
            maxclass = 'negative'
        elif (maxPosProb == maxNegProb and maxPosProb < maxNeutProb):
            maxclass = 'neutral'
        
        print (maxclass)

class NaiveBayesBinary:
    neg_dict = {}
    pos_dict = {}
    neut_dict = {}
    comm_dict = {}
    klass_freqs = {}
    totKlasses = 0
    pos_count = 0
    neg_count = 0
    neut_count = 0

    def __init__(self, klasses):
        self.train(klasses)
        line1 = 0
        for line2 in train_klasses:
            if line2 == 'positive':
                tokens = tokenize(train_texts[line1])
                tokens = set (tokens)
                for token in tokens:
                    self.pos_count = self.pos_count + 1
                    self.pos_dict[token] = self.pos_dict.get(token, 0) + 1
                    self.comm_dict[token] = self.comm_dict.get(token, 0) + 1
                line1 = line1 + 1
            elif ( line2 == 'negative'):
                tokens = tokenize(train_texts[line1])
                tokens = set (tokens)
                for token in tokens:
                    self.neg_count = self.neg_count + 1
                    self.neg_dict[token] = self.neg_dict.get(token, 0) + 1  
                    self.comm_dict[token] = self.comm_dict.get(token, 0) + 1
                line1 = line1 + 1
            elif ( line2 == 'neutral'):
                tokens = tokenize(train_texts[line1])
                tokens = set (tokens)
                for token in tokens:
                    self.neut_count = self.neut_count + 1
                    self.neut_dict[token] = self.neut_dict.get(token, 0) + 1  
                    self.comm_dict[token] = self.comm_dict.get(token, 0) + 1
                line1 = line1 + 1

    def train(self, klasses):
        # Count classes to determine which is the most frequent
        for k in klasses:
            self.totKlasses = self.totKlasses + 1
            self.klass_freqs[k] = self.klass_freqs.get(k, 0) + 1
        self.mfc = sorted(self.klass_freqs, reverse=True,key=lambda x : self.klass_freqs[x])[0]
    
    def classify(self, test_instance):
        maxPosProb = math.log10(self.klass_freqs.get('positive')/self.totKlasses)
        maxNegProb = math.log10(self.klass_freqs.get('negative')/self.totKlasses)
        maxNeutProb = math.log10(self.klass_freqs.get('neutral')/self.totKlasses)
        maxclass = 'neutral'
        tokens = tokenize(test_instance)
        tokens = set (tokens)

        for token in tokens:
            if ((self.pos_dict.get(token)!=None) and (self.neg_dict.get(token)!=None) and (self.neut_dict.get(token)!=None)):
                maxPosProb = maxPosProb + math.log10((self.pos_dict.get(token) + 1) / (self.pos_count + len(self.comm_dict)))
                maxNegProb = maxNegProb + math.log10((self.neg_dict.get(token) + 1) / (self.neg_count + len(self.comm_dict)))
                maxNeutProb = maxNeutProb + math.log10((self.neut_dict.get(token) + 1) / (self.neut_count + len(self.comm_dict)))

            elif (self.pos_dict.get(token)!=None and (self.neg_dict.get(token)!=None) and (self.neut_dict.get(token)==None)): 
                maxPosProb = maxPosProb + math.log10((self.pos_dict.get(token) + 1) / (self.pos_count + len(self.comm_dict)))
                maxNegProb = maxNegProb + math.log10((self.neg_dict.get(token) + 1) / (self.neg_count + len(self.comm_dict)))
                maxNeutProb = maxNeutProb + math.log10((0 + 1) / (self.neut_count + len(self.comm_dict)))

            elif(self.pos_dict.get(token)!=None and (self.neg_dict.get(token)==None) and (self.neut_dict.get(token)!=None)):
                maxPosProb = maxPosProb + math.log10((self.pos_dict.get(token) + 1) / (self.pos_count + len(self.comm_dict)))
                maxNegProb = maxNegProb + math.log10((0 + 1) / (self.neg_count + len(self.comm_dict)))
                maxNeutProb = maxNeutProb + math.log10((self.neut_dict.get(token) + 1) / (self.neut_count + len(self.comm_dict)))

            elif(self.pos_dict.get(token)!=None and (self.neg_dict.get(token)==None) and (self.neut_dict.get(token)==None)):
                maxPosProb = maxPosProb + math.log10((self.pos_dict.get(token) + 1) / (self.pos_count + len(self.comm_dict)))
                maxNegProb = maxNegProb + math.log10((0 + 1) / (self.neg_count + len(self.comm_dict)))
                maxNeutProb = maxNeutProb + math.log10((0 + 1) / (self.neut_count + len(self.comm_dict)))

            elif(self.pos_dict.get(token)==None and (self.neg_dict.get(token)!=None) and (self.neut_dict.get(token)!=None)):
                maxPosProb = maxPosProb + math.log10((0 + 1) / (self.pos_count + len(self.comm_dict)))
                maxNegProb = maxNegProb + math.log10((self.neg_dict.get(token) + 1) / (self.neg_count + len(self.comm_dict)))
                maxNeutProb = maxNeutProb + math.log10((self.neut_dict.get(token) + 1) / (self.neut_count + len(self.comm_dict)))
            
            elif(self.pos_dict.get(token)==None and (self.neg_dict.get(token)!=None) and (self.neut_dict.get(token)==None)):
                maxPosProb = maxPosProb + math.log10((0 + 1) / (self.pos_count + len(self.comm_dict)))
                maxNegProb = maxNegProb + math.log10((self.neg_dict.get(token) + 1) / (self.neg_count + len(self.comm_dict)))
                maxNeutProb = maxNeutProb + math.log10((0 + 1) / (self.neut_count + len(self.comm_dict)))

            elif(self.pos_dict.get(token)==None and (self.neg_dict.get(token)==None) and (self.neut_dict.get(token)!=None)):
                maxPosProb = maxPosProb + math.log10((0 + 1) / (self.pos_count + len(self.comm_dict)))
                maxNegProb = maxNegProb + math.log10((0 + 1) / (self.neg_count + len(self.comm_dict)))
                maxNeutProb = maxNeutProb + math.log10((self.neut_dict.get(token) + 1) / (self.neut_count + len(self.comm_dict)))

            elif(self.pos_dict.get(token)==None and (self.neg_dict.get(token)==None) and (self.neut_dict.get(token)==None)):
                maxPosProb = maxPosProb + math.log10((0 + 1) / (self.pos_count + len(self.comm_dict)))
                maxNegProb = maxNegProb + math.log10((0 + 1) / (self.neg_count + len(self.comm_dict)))
                maxNeutProb = maxNeutProb + math.log10((0 + 1) / (self.neut_count + len(self.comm_dict)))
        
        if (maxNeutProb > maxNegProb and maxNeutProb > maxPosProb):
            maxclass = 'neutral'
        elif (maxNegProb > maxPosProb and maxNegProb > maxNeutProb):
            maxclass = 'negative'
        elif (maxPosProb > maxNegProb and maxPosProb > maxNeutProb):
            maxclass = 'positive'
        elif (maxPosProb == maxNegProb and maxPosProb > maxNeutProb):
            maxclass = 'negative'
        elif (maxPosProb == maxNegProb and maxPosProb < maxNeutProb):
            maxclass = 'neutral'
        
        print (maxclass)

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # Method will be one of 'baseline', 'lr', 'lexicon', 'nb', or
    # 'nbbin'
   
    #method = sys.argv[1]
    #train_texts_fname = sys.argv[2]
    #train_klasses_fname = sys.argv[3]
    #test_texts_fname = sys.argv[4]

    method = 'nb'
    train_texts_fname = 'train.docs.txt'
    train_klasses_fname = 'train.classes.txt'
    test_texts_fname = 'dev.docs.txt'
    
    train_texts = [x.strip() for x in open(train_texts_fname,
                                           encoding='utf8')]
    train_klasses = [x.strip() for x in open(train_klasses_fname,
                                             encoding='utf8')]
    test_texts = [x.strip() for x in open(test_texts_fname,
                                          encoding='utf8')]
    pos_words = [x.strip() for x in open('pos-words.txt',
                                          encoding='utf8')]
    neg_words = [x.strip() for x in open('neg-words.txt',
                                          encoding='utf8')]
    
    if method == 'baseline':
        classifier = Baseline(train_klasses)
        results = [classifier.classify(x) for x in test_texts]

    if method == 'lexicon':
        classifier = Lexicon(train_klasses)
        results = [classifier.classify(x) for x in test_texts]

    if method == 'nb':
        classifier = NaiveBayes(train_klasses)
        results = [classifier.classify(x) for x in test_texts]
        
    if method == 'nbbin':
        classifier = NaiveBayesBinary(train_klasses)
        results = [classifier.classify(x) for x in test_texts]

    elif method == 'lr':
        # Use sklearn's implementation of logistic regression
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        
        count_vectorizer = CountVectorizer(analyzer=tokenize)

        train_counts = count_vectorizer.fit_transform(train_texts)

        lr = LogisticRegression(multi_class='multinomial',
                                solver='sag',
                                penalty='l2',
                                max_iter=1000,
                                random_state=0)
        clf = lr.fit(train_counts, train_klasses)
        
        test_counts = count_vectorizer.transform(test_texts)
        results = clf.predict(test_counts)
