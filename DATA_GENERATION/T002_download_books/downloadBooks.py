

from urllib.request import urlopen

filePath = "/home/zaid/DrPascal/data/NOVELS/"

url = "https://www.gutenberg.org/browse/scores/top"

page = urlopen(url)

html_bytes = page.read()
html = html_bytes.decode("utf-8")

#mainURL = "https://www.gutenberg.org"

