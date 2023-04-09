
@app.route("/test")
def test():
    sample_mds = [
        {
        "speaker" : "User",
        "markdown" : "Can you name some fruits?"
        },
        {
        "speaker" : "GLaDOS",
        "markdown" : """Of course! Here are some fruits.
* Apple: The fruit of the apple tree, which is native to North America.
* Apricot: The fruit of the apricot tree, which is native to South America.
* Banana: The fruit of the banana tree, which is native to Africa.
* Blackberry: The fruit of the blackberry tree, which is native to Asia.
* Cherry: The fruit of the cherry tree, which is native to North America.
* Fennel: The fruit of the fennel plant, which is native to Europe.
* Grapes: The fruit of the grapefruit tree, which is native to North America.
* Lemongrass: The fruit of the lemongrass plant, which is native to Southeast Asia.
* Mandarin: The fruit of the mandarin tree, which is native to China.
* Papaya: The fruit of the papaya tree, which is native to Asia.
* Peach: The fruit of the peach tree, which is native to North America.
* Pineapple: The fruit of the pineapple tree, which is native to Asia.
* Plums: The fruit of the plum tree, which is native to"""

        },
        {
        "speaker" : "User",
        "markdown" : "Write something in python!"
        },
        {
        "speaker" : "GLaDOS",
        "markdown" : """```
print("Hello World!")
```"""
        },
    ]
    for item in sample_mds:
        item["html"] = commonmark_to_html(item["markdown"])
    #speakers = [item["speaker"] for item in sample_mds]
    #html_blurbs = [ for item in sample_mds]

    return render_template("convo_2.html",  messages=sample_mds)