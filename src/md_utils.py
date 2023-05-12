import logging
import re
import pandoc

logger = logging.getLogger(__name__)

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
            sec = replace_newline_with_br(sec)
        fixed_sections.append(sec)
    updated_md = "```".join(fixed_sections)
    return updated_md


# TODO : Simplify this function
# Alternately train the model to output breaks on it's own
def identify_break_points(text):
    replace_spots = []
    line_so_far = ""
    skippable = False
    for i, char in enumerate(text):
        if char == "\n" and \
            (i > 0 and text[i-1]!= "\n") and \
            (i < len(text) - 1 and text[i+1]!= "\n") and \
            "|" not in line_so_far and \
            not skippable:
            replace_spots.append(i)
        if char != "\n":
            line_so_far += char
            stripped = line_so_far.strip()
            if len(stripped) > 0 and (not stripped[0].isalpha()):
                skippable = True
        else:
            line_so_far = ""
            skippable = False
    return replace_spots

def replace_newline_with_br(text):
    text = text.strip()
    replace_spots = identify_break_points(text)
    replace_spots.reverse()
    for i in replace_spots:
        text = text[:i] + "<br>\n" + text[i+1:]
    return text

def commonmark_to_html(md_text):
    """Convert the given markdown text to html with prettyprints for code blocks

    Args:
        md_text (str): Markdown text to convert

    Returns:
        str: HTML str output
    """
    md_text = fix_lines(md_text)
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