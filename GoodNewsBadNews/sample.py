from pattern.vector import SVM, KNN, NB, count, shuffled
from pattern.en     import tag, predicative


classifier = SVM()

classifier = SVM.load("sentiment.p")

def instance(review):                     # "Great book!"
    v = tag(review)                       # [("Great", "JJ"), ("book", "NN"), ("!", "!")]
    v = [word for (word, pos) in v if pos in ("JJ", "RB", "VB", "VBZ" ,"NN", "NNS", "NNP", "NNPS") or word in ("!")]
    v = [predicative(word) for word in v] # ["great", "!", "!"]
    v = count(v)                          # {"great": 1, "!": 1}
    return v

score = classifier.classify(instance("you little bitch"))

print(score)
