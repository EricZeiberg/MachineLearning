from pattern.db     import Datasheet
from pattern.en     import tag, predicative
from pattern.vector import SVM, KNN, NB, count, shuffled
import os
# import argparse

classifier = SVM()


print "loading data..."
data = os.path.join(os.path.dirname(__file__), "polarity2.csv")
data = Datasheet.load(data)
data = shuffled(data)

def instance(review):                     # "Great book!"
    v = tag(review)                       # [("Great", "JJ"), ("book", "NN"), ("!", "!")]
    v = [word for (word, pos) in v if pos in ("JJ", "RB", "VB", "VBZ", "NN", "NNS", "NNP", "NNPS") or word in ("!")]
    v = [predicative(word) for word in v] # ["great", "!", "!"]
    v = count(v)                          # {"great": 1, "!": 1}
    return v


# parser = argparse.ArgumentParser(description='This trains polarity data and then tests it on Reuters news!')
# parser.add_argument('-f', '--savefile', dest='savefile', default='checkpoint.p', help='file to save to: must have .p extension')
# args = parser.parse_args()
print "training..."
for score, review in data[:1000]:
    classifier.train(instance(review), type=int(score) > 0)
classifier.save("sentiment.p")

print "testing..."
i = n = 0
for score, review in data[1000:1500]:
    if classifier.classify(instance(review)) == (int(score) > 0):
        i += 1
    n += 1


print float(i) / n
