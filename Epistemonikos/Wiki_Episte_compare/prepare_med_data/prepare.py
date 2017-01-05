import xml.etree.ElementTree as ET

versions = "abcdefgh"
allmedline = ""
for letter in versions:
    tree = ET.parse("medsamp2016{}.xml".format(letter))
    root = tree.getroot()
    for neighbor in root.iter('AbstractText'):
        if neighbor.text:
            allmedline += neighbor.text

with open("../allmedline", "w") as _file:
    _file.write(allmedline)
