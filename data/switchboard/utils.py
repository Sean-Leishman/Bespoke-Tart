import re

OmitText = [
    "[silence]",
    "[noise]",
    "[vocalized-noise]",
]


def _clean_dialogs():
    pass


def _read_transcript_line(line):
    sepLine = line.split(" ")

    text = " ".join(sepLine[3:]).strip()
    start = float(sepLine[1])
    end = float(sepLine[2])

    return text, start, end


def _return_overlap(textA, textB, startA, startB, endA, endB):
    if startA > startB and endA < endB:
        return textA, 'A'
    elif startB > startA and endB < endA:
        return textB, 'B'

    return None, None


def _check_overlap_silence(past_line, next_line, thresh=1):
    past_text, past_start, past_end = _read_transcript_line(past_line)
    next_text, next_start, next_end = _read_transcript_line(next_line)

    if past_text != '[silence]' or next_text != '[silence]':
        return False

    if past_end - past_start < 1:
        return False

    if next_end - next_start < 1:
        return False

    return True


def read_txt(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    return lines

# Preprocessing handled by TurnGPT (https://github.com/ErikEkstedt/datasets_turntaking/blob/main/datasets_turntaking/dataset/switchboard/utils.py)


def sub_regex(s):
    """
    Switchboard annotation specific regexp.

    See:
        - `datasets_turntaking/features/dataset/switchboard.md`
        - https://www.isip.piconepress.com/projects/switchboard/doc/transcription_guidelines/transcription_guidelines.pdf

    """
    # Noise
    s = re.sub(r"\[noise\]", "", s)
    s = re.sub(r"\[vocalized-noise\]", "", s)

    # laughter
    s = re.sub(r"\[laughter\]", "", s)
    # laughing and speech e.g. [laughter-yeah] -> yeah
    s = re.sub(r"\[laughter-(\w*)\]", r"\1", s)
    s = re.sub(r"\[laughter-(\w*\'*\w*)\]", r"\1", s)

    # Partial words: w[ent] -> went
    s = re.sub(r"(\w+)\[(\w*\'*\w*)\]", r"\1\2", s)
    # Partial words: -[th]at -> that
    s = re.sub(r"-\[(\w*\'*\w*)\](\w+)", r"\1\2", s)

    # restarts
    s = re.sub(r"(\w+)-\s", r"\1 ", s)
    s = re.sub(r"(\w+)-$", r"\1", s)

    # Pronounciation variants
    s = re.sub(r"(\w+)\_\d", r"\1", s)

    # Mispronounciation [splace/space] -> space
    s = re.sub(r"\[\w+\/(\w+)\]", r"\1", s)

    # Coinage. remove curly brackets... keep word
    s = re.sub(r"\{(\w*)\}", r"\1", s)

    # remove double spacing on last
    s = re.sub(r"\s\s+", " ", s)
    return s.strip()  # remove whitespace start/end


def _extract_word_features(filename, speaker):
    def remove_multiple_whitespace(s):
        s = re.sub(r"\t", " ", s)
        return re.sub(r"\s\s+", " ", s)

    words = read_txt(filename)

    word_feats = {}
    for word_row in words:
        word_row = remove_multiple_whitespace(word_row).strip()

        key, start, end, word = word_row.split(" ")

        # Apply regex?
        word = sub_regex(word)

        # Check if word should be omitted
        if not (word in OmitText or word == ""):
            if key in word_feats:
                word_feats[key].append(
                    {
                        "word": word,
                        "start": float(start),
                        "end": float(end),
                    }
                )
            else:
                word_feats[key] = [{
                    "word": word,
                    "start": float(start),
                    "end": float(end),
                }]
    return word_feats


def extract_speaker_timings(transcript, min_word_diff=0.05):
    out = [[], []]
    for speaker in [0, 1]:
        for utterance in transcript[speaker]:
            start, end = utterance["wfeats"][0]["start"], utterance["wfeats"][-1]["end"]

            for word in utterance["wfeats"][1:]:
                if word["start"] - end < min_word_diff:
                    end = word["end"]
                else:
                    out[speaker].append((start, end))
                    start = word["start"]
                    end = word["end"]

            out[speaker].append((start, end))

    return out


def extract_dialog(filenames):
    trans_filenameA, words_filenameA, trans_filenameB, words_filenameB = filenames

    utterancesA = _extract_utterance_word_feats(
        trans_filenameA,
        words_filenameA,
        speaker='A'
    )
    utterancesB = _extract_utterance_word_feats(
        trans_filenameB,
        words_filenameB,
        speaker='B'
    )

    return [utterancesA, utterancesB]


def _extract_utterance_word_feats(trans_filename, words_filename, speaker):
    word_feats = _extract_word_features(words_filename, speaker)

    transcript = read_txt(trans_filename)

    utterances = []
    for row in transcript:
        key, start, end, *words = row.split(" ")

        if not (words[0] in OmitText and len(words) == 1):
            word_feat = word_feats.get(key, None)

            if word_feat is None:
                continue

            words = " ".join(words)

            # Apply regex?
            words = sub_regex(words)

            utterances.append(
                {
                    "text": words,
                    "wfeats": word_feat,
                    "start": word_feat[0]["start"],
                    "end": word_feat[-1]["end"],
                }
            )
    return utterances
