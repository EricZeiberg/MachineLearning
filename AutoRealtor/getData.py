from bs4 import BeautifulSoup
import urllib

path = "images/"

classMap = {"1": "Ranch",
            "2": "Raised Ranch",
            "3": "Split Level",
            "4": "Cape",
            "5": "Colonial",
            "6": "Contemporary",
            "7": "Mansion",
            "8": "Multi Family",
            "9": "Duplex",
            "10": "Condo",
            "11": "Tudor",
            "12": "Victorian",
            "13": "High Ranch",
            "14": "Apartments, Comm",
            "15": "Tri-Level"  }

def parsePage(pid):
    try:
        page = urllib.urlopen('http://gis.vgsi.com/westhartfordct/Parcel.aspx?pid=' + pid).read()
        soup = BeautifulSoup(page, "lxml")
        soup.prettify()
        img = soup.find_all("img")[2]
        imgURL = img["src"]
        if imgURL == "http://images.vgsi.com/photos/WestHartfordCTPhotos//default.jpg":
            return
        table = soup.find_all(id="MainContent_ctl01_panView")[0]
        row = table.find_all(class_="RowStyle")[0]
        houseType = row.find_all("td")[1]
        idValue = 16
        for classId, name in classMap.iteritems():
            if name == houseType.get_text():
                idValue = classId
        computedPath = path + pid + "_" + idValue + '.jpg'
        urllib.urlretrieve(imgURL, computedPath)
        print(computedPath)
    except Exception as e:
        print("exception raised " + e.message)
        return


for x in range(2, 300):
    parsePage(str(x))
