import codecs

f = codecs.open("gutenbergtop.html", 'r')
html = f.read()

print(type(html))

start_index = html.find("<ol>")
end_index = html.find("</ol>")


print(html)

print(start_index)
print(end_index)


