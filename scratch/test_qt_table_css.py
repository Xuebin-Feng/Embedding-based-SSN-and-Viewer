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
    background-color: #eff1f3; /* slightly darker grey */
    color: #1f2328;
}
table {
    border-collapse: collapse;
    border: 1px solid #d0d7de;
}
th {
    background-color: #f6f8fa;
    border: 1px solid #d0d7de;
    font-weight: bold;
    padding: 6px 10px;
}
td {
    border: 1px solid #d0d7de;
    padding: 6px 10px;
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
# Just add border="1" and cellpadding="6" to the table tag
html = html.replace("<table>", '<table border="1" cellpadding="6" style="border-collapse: collapse;">')

tb.setHtml(html)
tb.resize(600, 400)

# Render to QImage and save
img = QImage(QSize(600, 400), QImage.Format.Format_ARGB32)
img.fill(0xffffffff)
painter = QPainter(img)
tb.render(painter)
painter.end()

img.save("e:/OneDrive - University of Toronto/Research Records/Computational Methods/Sequence_Similarity_Network_Viewer/scratch/render_css.png")
print("Render saved to scratch/render_css.png")
