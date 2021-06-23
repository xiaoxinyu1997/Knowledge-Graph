# Relation Recognition


data_path = '../data/'
# Subgraph Extraction: extract the subgraphs within its two-hop 
# in the knowledge base, and each relationships in the subgraph 
# could be the target relationship.
def Subgraph(entity):
    file = open(data_path + 'triples.txt', 'r+', encoding = 'utf-8')
    triples = []
    while True:
        line = file.readline()
        if line:
            if line.find(entity) >= 1:
                # triple = ()
                bowls = line.split('\t')
                # triples.append(triple)
                elements = []
                elements.append(bowls[0][1:-1]) 
                elements.append(bowls[1][1:-1]) 
                elements.append(bowls[2][1:-3]) 
                triples.append(elements)
        else:
            break
    file.close()
    return triples

# Scoring Module: After getting allthe candidate relationships, 
# we constructed a scoring strategy.
def scoring(question, relations):
    # Score_relation_similarity
    s1 = score_relation_similarity(question, relations)
    # Score_object_similarity
    # Score_char_overlap

def score_relation_similarity():

# def score_object_similarity():

# def score_char_overlap():


question = '贴心流量券我为啥办理不了呀，怎么开通'

result = Subgraph('1元5GB流量券')
print(result)