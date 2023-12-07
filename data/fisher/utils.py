
def fisher_regexp(s, remove_restarts=False):
    """
    See information about annotations at:
    * https://catalog.ldc.upenn.edu/docs/LDC2004T19/fe_03_readme.txt
    Regexp
    ------
    * Special annotations:  ["[laughter]", "[noise]", "[lipsmack]", "[sigh]"]
    * double paranthesis "((...))" was not heard by the annotator
      and can be empty but if not empty the annotator made their
      best attempt of transcribing what was said.
    * What's "[mn]" ? (can't find source?)
        * Inaudible
        * seems to be backchannel or laughter
        * oh/uh-huh/mhm/hehe
    * Names/accronyms (can't find source?)
        * t._v. = TV
        * m._t._v. = MTV
    """

    # Noise
    s = re.sub(r"\[noise\]", "", s)
    # laughter
    s = re.sub(r"\[laughter\]", "", s)
    # lipsmack
    s = re.sub(r"\[lipsmack\]", "", s)
    # sigh
    s = re.sub(r"\[sigh\]", "", s)
    # [mn] inaubible?
    s = re.sub(r"\[mn\]", "", s)

    # clean restarts
    # if remove_restarts=False "h-" -> "h"
    # if remove_restarts=True  "h-" -> ""
    if remove_restarts:
        s = re.sub(r"(\w+)-\s", " ", s)
        s = re.sub(r"(\w+)-$", r"", s)
    else:
        s = re.sub(r"(\w+)-\s", r"\1 ", s)
        s = re.sub(r"(\w+)-$", r"\1", s)

    # doubble paranthesis (DP) with included words
    # sometimes there is DP inside another DP
    s = re.sub(r"\(\(((.*?)+)\)\)", r"\1", s)
    s = re.sub(r"\(\(((.*?)+)\)\)", r"\1", s)

    # empty doubble paranthesis
    s = re.sub(r"\(\(\s*\)\)", "", s)

    # Names/accronyms
    s = re.sub(r"\.\_", "", s)

    # remove punctuation
    # (not included in annotations but artifacts from above)
    s = re.sub(r"\.", "", s)

    # remove double spacing on last
    s = re.sub(r"\s\s+", " ", s)
    return s.strip()  # remove whitespace start/end

def read_txt(path:str):
    with open(path) as f:
        lines = f.readlines()
    return lines

def load_dialog(path: str):
    anno = [[], []]

    for row in read_text(path):
        if row == "":
            continue

        split_row = row.split(" ")
        if split_row[0] == "#":
            continue

        start = float(split_row[0])
        end = float(split_row[1])
        speaker = split_row[2].replace(":", "")
        channel = 0 if speaker == 'A' else 1
        text = " ".join(split_row[3:])

        text = regexp(text, remove_restarts=True)
        if len(text) == 0:
            continue

        anno[channel].append({
            "start": start,
            "end": end,
            "text": text,
        })
    return anno
