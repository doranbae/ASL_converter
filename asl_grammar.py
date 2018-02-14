# Trained POS tagger was too big to uploade on Github. 
# The following script relies on the POS tagger.

class eng2asl():
    def __init__(self,english_sentence):
        self.english_sentence = english_sentence
        assert isinstance(self.english_sentence, Iterable)
    
    def pos_tagger(self):
        self.pos_tagged = pos_tag(word_tokenize(self.english_sentence))
        self.token_n = len(self.pos_tagged)
        
    def create_matrix(self):
        self.matrix = np.zeros([self.token_n,3])
        
    def extract_time(self):
        self.ioe_tagged = chunker_model.parse(self.pos_tagged)
        for idx,i in enumerate(self.ioe_tagged):
            if len(i) > 1:
                pass
            else: 
                value = str(i)
                if value[:4] == '(tim':
                    self.matrix[idx,0] +=2    
        
    def extract_topic(self):
        self.chunked = sen_chunker_model.parse(self.pos_tagged)
        vp_list = []
        for idx, j in enumerate(self.chunked):
            try:
                if j.label() == 'VP':
                    vp_item = list((idx,j))
                    vp_list.append(vp_item)
            except:
                pass

        topic = vp_list[0]
        topic_idx = topic[0]
        topic_phrase = topic[1]
        for i in range(len(topic_phrase)):
            self.matrix[topic_idx + i ,1] +=1 
        
    def vis_matrix(self):
        df = pd.DataFrame(data=self.matrix) #, columns = {"topic_score","time_score", "etc_score"}
        df_headings = self.pos_tagged
        df["words"] = pd.Series(df_headings).values
        df = df.rename(columns = {0: "time_score", 1:"topic_score", 2:"etc_score"})
        df["total_score"] = df[["time_score", "topic_score", "etc_score"]].sum(axis=1) 
        print(df)
        
        
    def sum_score(self):
        self.word_order = np.sum(self.matrix, axis = 1)
        return self.word_order, self.english_sentence




class Convert_pictogram():
    def __init__(self,asl_sent_order, eng_sent):
        self.asl_sent_order = asl_sent_order
        self.eng_sent = eng_sent
        assert isinstance(self.asl_sent_order, Iterable)
    
    def arrange_sentence(self):
        asl_sent_order_tmp = list(self.asl_sent_order)
        sent_order_ = []
        for i in range(len(asl_sent_order_tmp)):
            if sum(asl_sent_order_tmp) == -1 * float(len(self.eng_sent)):
                break
            else:
                index, value = max(enumerate(asl_sent_order_tmp), key=operator.itemgetter(1))
                sent_order_.append(index)
                asl_sent_order_tmp[index] = - 1
        
        self.pos_tagged = pos_tag(word_tokenize(self.eng_sent))
        self.new_order_sent = [ self.pos_tagged[int(i)] for i in sent_order_]
    
    def filter_words(self):
        be_verb_set = {'be','being','am','are','was','were','is'}
        article_set = {'the','a', 'A'}
        i_set = {"I", "we", "my"}
        preposition_set = {'in','on'}
        posession_set = {"my"}
        for part in self.new_order_sent:
            wd = part[0]
            pos = part[1]
            if ((wd in be_verb_set) & (str(pos) == 'VBZ')):
                self.new_order_sent.remove(part)
            if ((wd in be_verb_set) & (str(pos) == 'VBP')):
                self.new_order_sent.remove(part)
            if ((wd in article_set) & (pos == 'DT')):
                self.new_order_sent.remove(part)
            if ((wd in i_set) & (pos == 'PRP')):
                self.new_order_sent.remove(part)
            if ((wd in posession_set) & (pos == 'PRP$')):
                self.new_order_sent.remove(part)
            if ((wd in preposition_set) & (pos == 'IN')):
                self.new_order_sent.remove(part)
        return self.new_order_sent