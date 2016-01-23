from pattern.vector import Document, Model, IG, TF, TFIDF, BINARY
import sys
import os

print "Reading sample code and instantiating documents..."
documents = []
exampleDir = "examples/"
for file in os.listdir(exampleDir):
    if os.path.isdir(exampleDir + file):
        for subfile in os.listdir(exampleDir + file):
            if (os.path.isfile(exampleDir + file + "/" + subfile)):
                with open (exampleDir + file + "/" + subfile, "r") as langDoc:
                    text = langDoc.read()
                    doc = Document(text, type=file)
                    documents.append(doc)

print "Creating statistical model..."
m = Model(documents=documents, weight=IG)

# Test with sample Java doc
print "Comparing test document..."
with open ("coffee.txt", "r") as myfile:
    testFile = myfile.read()
testDoc = Document(testFile, type='Java')
testSimilarities = m.neighbors(testDoc, top=10)
prediction = testSimilarities[0][1].type #neighbors() returns (similarity, document) list
confidence = testSimilarities[0][0]
print "LanguageLearn has predicted " + testSimilarities[0][1].type + " with a " + str(round(confidence * 100, 2)) + "% confidence"
