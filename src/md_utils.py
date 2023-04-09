def fix_lines(base_md):
    sections = base_md.split("```")
    fixed_sections = []
    for i, sec in enumerate(sections):
        if i % 2 == 0:
            sec = sec.replace("\n", "\n\n")
        fixed_sections.append(sec)
    return "```".join(fixed_sections)