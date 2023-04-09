def fix_lines(base_md):
    """Doubles newlines outside of code blocks to fix formatting issue from model training code.

    Args:
        base_md (str): Markdown to update

    Returns:
        [type]: [description]
    """
    # TODO : Use regex and only add one extra newline to single and double newlines, leave rest unaffected
    sections = base_md.split("```")
    fixed_sections = []
    for i, sec in enumerate(sections):
        if i % 2 == 0:
            sec = sec.replace("\n", "\n\n")
        fixed_sections.append(sec)
    return "```".join(fixed_sections)


def commonmark_to_html(md_text):
    """Convert the given markdown text to html with prettyprints for code blocks

    Args:
        md_text (str): Markdown text to convert

    Returns:
        str: HTML str output
    """
    pd_data = pandoc.read(md_text, format="gfm")
    html_data = pandoc.write(pd_data, format="html")
    html_data = re.sub("<code[^>]*>", '<code class="prettyprint">', html_data)
    
    return html_data

def htmlify_convo(convo, speakers=("User", "GLaDOS")):
    md_messages = []
    for idx, message in enumerate(convo):
        d = {"speaker" : speakers[idx % 2], "html" : commonmark_to_html(message)}
        md_messages.append(d)
    return md_messages

def basicify_convo(convo,  speakers=("User", "GLaDOS")):
    md_messages = []
    for idx, message in enumerate(convo):
        message = message.replace("\n", "<br>")
        d = {"speaker" : speakers[idx % 2], "html" : f'<code>{message}</code>'}
        md_messages.append(d)
    return md_messages