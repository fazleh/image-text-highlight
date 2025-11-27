from flask import Flask, render_template_string, send_from_directory, abort
import os

app = Flask(__name__)

# CHANGE THIS: folder you want to expose
BASE_DIR = "/data/image-text-highlight/documents/"   # example path on your server


# HTML template (inline for simplicity)
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>File Browser</title>
    <style>
        body { font-family: Arial; margin: 40px; }
        table { width: 600px; border-collapse: collapse; }
        th, td { padding: 8px; border-bottom: 1px solid #ddd; }
        a { text-decoration: none; color: #0645AD; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h2>Index of /{{ path }}</h2>
    <table>
        <tr><th>Name</th><th>Type</th></tr>
        {% if parent %}
        <tr>
            <td><a href="{{ parent }}">.. (parent)</a></td>
            <td>dir</td>
        </tr>
        {% endif %}
        {% for name, type_, link in items %}
        <tr>
            <td><a href="{{ link }}">{{ name }}</a></td>
            <td>{{ type_ }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""


def safe_join(base, *paths):
    """Prevent path traversal attacks."""
    final_path = os.path.realpath(os.path.join(base, *paths))
    if not final_path.startswith(os.path.realpath(base)):
        abort(403)
    return final_path


@app.route("/", defaults={"req_path": ""})
@app.route("/<path:req_path>")
def browse(req_path):
    abs_path = safe_join(BASE_DIR, req_path)

    if os.path.isfile(abs_path):
        directory = os.path.dirname(abs_path)
        filename = os.path.basename(abs_path)
        return send_from_directory(directory, filename, as_attachment=True)

    # List directory contents
    files = []
    for name in os.listdir(abs_path):
        full = os.path.join(abs_path, name)
        rel = os.path.join(req_path, name)
        if os.path.isdir(full):
            files.append((name + "/", "dir", "/" + rel))
        else:
            files.append((name, "file", "/" + rel))

    parent = None
    if req_path:
        parent = "/" + os.path.dirname(req_path)

    return render_template_string(
        TEMPLATE,
        items=files,
        path=req_path,
        parent=parent
    )


if __name__ == "__main__":
    # Run on all interfaces, port 8000
    app.run(host="0.0.0.0", port=8000, debug=False)

