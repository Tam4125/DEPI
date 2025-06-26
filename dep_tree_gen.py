import spacy
import math



# Generate dependency tree for single sentence
def tree_generate(sentence, processor):
    doc = processor(sentence)
    return [(token.i, token.dep_, token.head.i) for token in doc]


def string_compare(str1, str2):
    return 1 if str1 == str2 else 0

def relation_compare(re1, re2, alpha):
    return alpha if re1 == re2 else 1


def dep_matrix_gen(sentence1, sentence2, processor, alpha):
    dep1 = tree_generate(sentence1, processor)
    dep2 = tree_generate(sentence2, processor)
    n, m = len(dep1), len(dep2)

    words1 = [token.text for token in processor(sentence1)]
    words2 = [token.text for token in processor(sentence2)]

    dep_matrix = [[0 for _ in range(n+m)] for _ in range(n+m)]


    # Between two sentences
    for i in range(n):
        tri1 = dep1[i]
        dep_matrix[tri1[0]][tri1[2]] = 1
        for j in range(m):
            tri2 = dep2[j]
            dep_matrix[tri2[0]+n][tri2[2]+n] = 1

            tmp = string_compare(words1[tri1[0]], words2[tri2[0]]) + string_compare(words1[tri1[2]], words2[tri2[2]])
            tmp = tmp * relation_compare(tri1[1], tri2[1], alpha)

            dep_matrix[i][n+j] = tmp
            dep_matrix[n+j][i] = tmp    
    
    return dep_matrix

# Load the spaCy English model
processor = spacy.load("en_core_web_sm")
sentence1 = "I love apples"
sentence2 = "You enjoy apples"
print(dep_matrix_gen(sentence1, sentence2, processor, 2))







