from flask import Flask, render_template
from SOWebsite.SOWebsite.Labs import TagLab, QuestionLab

app = Flask(__name__)
qlab = QuestionLab()
tlab = TagLab()
frameworks = [{"href": "/caffe", "caption": "Caffe"},
                  {"href": "/tensorflow", "caption": "TensorFlow"},
                  {"href": "/keras", "caption": "Keras"},
                  {"href": "torch", "caption": "Torch"},
                  {"href": "theano", "caption": "Theano"},
                  {"href": "lasagne", "caption": "Lasagne"},
                  {"href": "mxnet", "caption": "Mxnet"}]


@app.route('/')
def overview():
    return render_template("overview.html", frameworks=frameworks)


@app.route("/<fmw>")
def framework(fmw):
    faq = [{"qname": str(q)[:str(q).rfind("@") - 2].lstrip("<Question '"), "qlink": q.link} for q in qlab.get_faq("{}faq".format(fmw), fmw)]
    wiki_string = "http://stackoverflow.com/tags/{}/info"
    rtags = [{"tname": t.name, "twiki": wiki_string.format(t.name)} for t in tlab.get_related(fmw, "{}related".format(fmw))]
    return render_template("framework.html", frameworks=frameworks, faq=faq, rtags=rtags)


@app.route("/faq/caffe")
def faq():
    s = ""
    for q in qlab.get_faq("caffeafaq", "caffe"):
        s += str(q).lstrip("<").rstrip(">")
    return s.lstrip("<").rstrip(">")


if __name__ == '__main__':
    app.run()
