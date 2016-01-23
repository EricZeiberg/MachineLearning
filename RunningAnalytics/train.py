from bs4 import BeautifulSoup, NavigableString, Tag
import numpy as np
from sklearn import cross_validation
from sklearn.linear_model import Ridge

def parseHTML(html):
    scaleArray = []
    sleepArray = []
    soup = BeautifulSoup(html, "html.parser")
    logs = soup.find_all("form")[0]
    i = 0
    for t in logs.find_all("table", class_="encapsule"):
        if i == 0:
            i+=1 # First table node is something else, so skip the first result of array
            continue
        comments = t.find_all("td", class_="pad-reset")[0].find_all("tr")[3].find_all("td")[1]
        containsScale = False
        for br in comments.find_all("br"):
            next = br.nextSibling # Next few lines split the comment by <br> tags
            if not (next and isinstance(next, NavigableString)):
                continue
            next2 = next.nextSibling
            if next2 and isinstance(next2,Tag) and next2.name == 'br':
                text = str(next).strip()
                if text:
                    if "SCALE: " in text:
                        scaleArray.append(int(text.split(": ")[1]))
                        containsScale = True
                        break

        if containsScale == False:
            continue # If no scale data found in comment, skip finding sleep and go to next entry
        sleepHTML = t.find_all("td", class_="pad-reset")[0].find_all("div")[1].find_all("div")[0].find_all("tr")[1].find_all("td")[1]
        sleep = sleepHTML.text
        sleep = sleep.strip()
        if sleep == "--":
            sleepArray.append(sum(sleepArray)/float(len(sleepArray))) # If no sleep entered for that day, fill in with mean of all collected sleep data so far
            continue

        sleepArray.append(float(sleep.split(" ")[0]))


    sleepArray = np.reshape(sleepArray, (len(sleepArray), 1)) # Sklearn lib likes data in a [data, 1] format, so this reshapes the array
    scaleArray = np.reshape(scaleArray, (len(scaleArray), 1))

    return [sleepArray, scaleArray]

html = ""
with open ("R2WTest.html", "r") as myfile: # Downloaded the HTML from Running2Win for 2014-2015 as training data
    html = myfile.read()

data = parseHTML(html)


ridge = Ridge() # Tried to fit this data with a couple other regression modules, this one gave the best result
ridge.fit(data[0], data[1])

test = ""
with open ("R2Win2015.html", "r") as myfile: # HTML from 2015-2016 used as test data to calculate model score
    test = myfile.read()

testData = parseHTML(test)
#print(ridge.score(testData[0], testData[1])) - Returns Coefficient of determination as very close to 0, so model fits very well
print ridge.predict(10) # 7.27747915
print ridge.predict(1) # 7.19472833
