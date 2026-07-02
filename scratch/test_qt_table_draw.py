import sys
import markdown
from PyQt6.QtWidgets import QApplication, QTextBrowser
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtCore import QSize

app = QApplication(sys.argv)

tb = QTextBrowser()
tb.setReadOnly(True)

github_style = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 13px;
    line-height: 1.5;
    color: #24292e;
    background-color: #ffffff;
}
code {
    font-family: monospace;
    background-color: #f6f8fa;
    color: #1f2328;
}
table {
    border-collapse: collapse;
}
th {
    background-color: #f6f8fa;
}
"""

tb.document().setDefaultStyleSheet(github_style)

md = """
# Test
This is some `inline_code`.

| Param | Desc |
| :--- | :--- |
| `INPUT_FASTA` | Description here |
"""

html = markdown.markdown(md, extensions=['tables'])
html = html.replace("<table>", '<table border="1" cellpadding="6" style="border-collapse: collapse; border: 1px solid #d0d7de;">')
html = html.replace("<th>", '<th style="border: 1px solid #d0d7de; background-color: #f6f8fa; font-weight: bold;">')
html = html.replace("<td>", '<td style="border: 1px solid #d0d7de;">')

tb.setHtml(html)
tb.resize(600, 400)

# Render to QImage and save
img = QImage(QSize(600, 400), QImage.Format.Format_ARGB32)
img.fill(0xffffffff) # white background
painter = QPainter(img)
tb.render(painter)
painter.end()

img.save("e:/OneDrive - University of Toronto/Research Records/Computational Methods/Sequence_Similarity_Network_Viewer/scratch/render.png")
print("Render saved to scratch/render.png")
