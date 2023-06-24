from typing import Union, List, Optional
import warnings
import torch as t
import numpy as np
from transformers import AutoTokenizer
import random
import copy
import re

NAMES = [
    "Aaron",
    "Adam",
    "Alan",
    "Alex",
    "Alice",
    "Amy",
    "Anderson",
    "Andre",
    "Andrew",
    "Andy",
    "Anna",
    "Anthony",
    "Arthur",
    "Austin",
    "Blake",
    "Brandon",
    "Brian",
    "Carter",
    "Charles",
    "Charlie",
    "Christian",
    "Christopher",
    "Clark",
    "Cole",
    "Collins",
    "Connor",
    "Crew",
    "Crystal",
    "Daniel",
    "David",
    "Dean",
    "Edward",
    "Elizabeth",
    "Emily",
    "Eric",
    "Eva",
    "Ford",
    "Frank",
    "George",
    "Georgia",
    "Graham",
    "Grant",
    "Henry",
    "Ian",
    "Jack",
    "Jacob",
    "Jake",
    "James",
    "Jamie",
    "Jane",
    "Jason",
    "Jay",
    "Jennifer",
    "Jeremy",
    "Jessica",
    "John",
    "Jonathan",
    "Jordan",
    "Joseph",
    "Joshua",
    "Justin",
    "Kate",
    "Kelly",
    "Kevin",
    "Kyle",
    "Laura",
    "Leon",
    "Lewis",
    "Lisa",
    "Louis",
    "Luke",
    "Madison",
    "Marco",
    "Marcus",
    "Maria",
    "Mark",
    "Martin",
    "Mary",
    "Matthew",
    "Max",
    "Michael",
    "Michelle",
    "Morgan",
    "Patrick",
    "Paul",
    "Peter",
    "Prince",
    "Rachel",
    "Richard",
    "River",
    "Robert",
    "Roman",
    "Rose",
    "Ruby",
    "Russell",
    "Ryan",
    "Sarah",
    "Scott",
    "Sean",
    "Simon",
    "Stephen",
    "Steven",
    "Sullivan",
    "Taylor",
    "Thomas",
    "Tyler",
    "Victoria",
    "Warren",
    "William",
]

ABC_TEMPLATES = [
    "Then, [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
    "Afterwards [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
    "When [A], [B] and [C] arrived at the [PLACE], [B] and [C] gave a [OBJECT] to [A]",
    "Friends [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
]

BAC_TEMPLATES = [
    template.replace("[B]", "[A]", 1).replace("[A]", "[B]", 1)
    for template in ABC_TEMPLATES
]

BABA_TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

BABA_LONG_TEMPLATES = [
    "Then in the morning, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After taking a long break [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When soon afterwards [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When soon afterwards [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While spending time together [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While spending time together [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch in the afternoon, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, while spending time together [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then in the morning afterwards, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The local big [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends separated at birth [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

BABA_EXPANDED_TEMPLATES = [
"Then in the morning, [B] along with [A] went directly to the [PLACE]. [B] gave a beautiful [OBJECT] to [A]",
"Then in the morning, [B] along with [A] enjoyed a day full of fun at the [PLACE]. In the end, [B] gave a lovely [OBJECT] to [A]",
"Then in the morning, [B] in collaboration with [A], were industriously working at the [PLACE]. Subsequently, [B] decided to give a precious [OBJECT] to [A]",
"Then in the morning, [B] and [A] after contemplating their options, thought about visiting the [PLACE]. [B] expressed his desire to give an interesting [OBJECT] to [A]",
"Then in the morning, following a lengthy, heated argument between [B] and [A] , [B] finally said to [A]",
"After a rejuvenating long break, [B] with [A] went to the [PLACE]. There, [B] gave a unique [OBJECT] to [A]",
"When soon afterwards, [B] and [A] much to their surprise, found a [OBJECT] at the [PLACE], [B] instantly decided to give it to [A]",
"When soon afterwards, [B] and [A] happened upon a [OBJECT] at the [PLACE], [B] made the choice to give the [OBJECT] to [A]",
"While enjoying their shared time, [B] and [A] were diligently working at the [PLACE]. In that moment, [B] gave a helpful [OBJECT] to [A]",
"While making the most of their shared time, [B] and [A] were commuting to the [PLACE]. En route, [B] handed over a handy [OBJECT] to [A]",
"After a hearty lunch in the afternoon, [B] accompanied by [A] went to the [PLACE]. [B] then gave a thoughtful [OBJECT] to [A]",
"Afterwards, while cherishing their time spent together, [B] and [A] decided to go to the [PLACE]. There, [B] gave a sentimental [OBJECT] to [A]",
"Then in the morning, after a long, drawn-out argument between [B] and [A] . [B] eventually said to [A]",
"The famous local [PLACE] that [B] and [A] visited had an intriguing [OBJECT]. [B] ended up giving it to [A]",
"Long-lost friends, [B] and [A] discovered a hidden [OBJECT] at the [PLACE]. In a moment of kindness, [B] gave it to [A]",
]

BABA_COMPLEMENT_TEMPLATES = [
"Then, [B] and [A] together went to the [PLACE]. There, [B] gave a [OBJECT] to [A]",
"Then, [B] and [A] spent an enjoyable time at the [PLACE]. There, [B] gave a [OBJECT] to [A]",
"Then, [B] and [A] were seen working at the [PLACE]. It was there that [B] decided to give a [OBJECT] to [A]",
"Then, [B] and [A] contemplated going to the [PLACE]. It was then [B] wanted to give a [OBJECT] to [A]",
"Then, [B] and [A] had a long argument, after which [B] said to [A]",
"After [B] and [A] visited the [PLACE], it was there that [B] gave a [OBJECT] to [A]",
"When [B] and [A] got a [OBJECT] at the [PLACE], that was when [B] decided to give it to [A]",
"When [B] and [A] got a [OBJECT] at the [PLACE], it was then that [B] decided to give the [OBJECT] to [A]",
"While [B] and [A] were busy working at the [PLACE], it happened that [B] gave a [OBJECT] to [A]",
"While [B] and [A] were commuting to the [PLACE], it was during this time that [B] gave a [OBJECT] to [A]",
"After the lunch, [B] and [A] made their way to the [PLACE]. At that moment, [B] gave a [OBJECT] to [A]",
"Afterwards, [B] and [A] visited the [PLACE]. It was there that [B] gave a [OBJECT] to [A]",
"Then, [B] and [A] had a long argument. Once they had finished, [B] said to [A]",
"The [PLACE] [B] and [A] decided to visit had a [OBJECT]. [B] decided to give it to [A]",
"Friends [B] and [A] discovered a [OBJECT] at the [PLACE]. That's when [B] gave it to [A]",
]


BABA_EXTRA_AND_TEMPLATES = [
"Then [B] along with [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
"Then [B] in the company of [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
"Then [B] joined by [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
"Then [B] together with [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
"Then [B] in a heated discussion with [A] had a long argument. Afterwards, [B] said to [A]",
"After [B] accompanied by [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
"When [B] along with [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
"When [B] joined by [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
"While [B] accompanied by [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
"While [B] in the company of [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
"After the lunch [B] along with [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
"Afterwards [B] accompanied by [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
"Then [B] after arguing with [A] had a long argument. Afterwards, [B] said to [A]",
"The [PLACE] that [B] in the company of [A] went to had a [OBJECT]. [B] gave it to [A]",
"Friends [B] and [A] on an adventurous journey found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]


BABA_EXTRA_AND2_TEMPLATES = [
    "Then, [B] and and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

BABA_NAME_B_PREFIX_TEMPLATES = [
" Benjamin had a good night's sleep. Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
" Benjamin had breakfast. Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
" Benjamin read the news. Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
" Benjamin got dressed. Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
" Benjamin checked the mail. Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
" Benjamin finished a book. After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
" Benjamin made a call. When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
" Benjamin did some exercises. When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
" Benjamin took a shower. While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
" Benjamin prepared a meal. While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
" Benjamin wrote a letter. After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
" Benjamin watered the plants. Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
" Benjamin swept the floor. Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
" Benjamin fed the cat. The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
" Benjamin set an alarm. Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

BABA_NAME_A_PREFIX_TEMPLATES = [
" Benjamin had a good night's sleep. Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
" Benjamin had breakfast. Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
" Benjamin read the news. Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
" Benjamin got dressed. Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
" Benjamin checked the mail. Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
" Benjamin finished a book. After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
" Benjamin made a call. When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
" Benjamin did some exercises. When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
" Benjamin took a shower. While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
" Benjamin prepared a meal. While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
" Benjamin wrote a letter. After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
" Benjamin watered the plants. Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
" Benjamin swept the floor. Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
" Benjamin fed the cat. The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
" Benjamin set an alarm. Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [Ad]",
]



ABBA_TEMPLATES = BABA_TEMPLATES[:]
ABBA_LONG_TEMPLATES = BABA_LONG_TEMPLATES[:]
ABBA_EXPANDED_TEMPLATES = BABA_EXPANDED_TEMPLATES[:]
ABBA_COMPLEMENT_TEMPLATES = BABA_COMPLEMENT_TEMPLATES[:]
ABBA_EXTRA_AND_TEMPLATES = BABA_EXTRA_AND_TEMPLATES[:]
ABBA_NAME_B_PREFIX_TEMPLATES = BABA_NAME_B_PREFIX_TEMPLATES[:]
ABBA_NAME_A_PREFIX_TEMPLATES = BABA_NAME_A_PREFIX_TEMPLATES[:]
ABBA_EXTRA_AND2_TEMPLATES = BABA_EXTRA_AND2_TEMPLATES[:]

for TEMPLATES in [ABBA_TEMPLATES, ABBA_LONG_TEMPLATES, ABBA_EXPANDED_TEMPLATES, ABBA_COMPLEMENT_TEMPLATES, 
                  ABBA_EXTRA_AND_TEMPLATES, ABBA_NAME_B_PREFIX_TEMPLATES, ABBA_NAME_A_PREFIX_TEMPLATES, ABBA_EXTRA_AND2_TEMPLATES]:
    for i in range(len(TEMPLATES)):
        first_clause = True
        for j in range(1, len(TEMPLATES[i]) - 1):
            if TEMPLATES[i][j - 1 : j + 2] == "[B]" and first_clause:
                TEMPLATES[i] = TEMPLATES[i][:j] + "A" + TEMPLATES[i][j + 1 :]
            elif TEMPLATES[i][j - 1 : j + 2] == "[A]" and first_clause:
                first_clause = False
                TEMPLATES[i] = TEMPLATES[i][:j] + "B" + TEMPLATES[i][j + 1 :]

VERBS = [" tried", " said", " decided", " wanted", " gave"]

PLACES = [
    "store",
    "garden",
    "restaurant",
    "school",
    "hospital",
    "office",
    "house",
    "station",
]

OBJECTS = [
    "ring",
    "kiss",
    "bone",
    "basketball",
    "computer",
    "necklace",
    "drink",
    "snack",
]


def gen_prompt_uniform(
    templates, names, nouns_dict, N, symmetric, prefixes=None, abc=False
):
    nb_gen = 0
    ioi_prompts = []
    while nb_gen < N:
        temp = random.choice(templates)
        temp_id = templates.index(temp)
        name_1 = ""
        name_2 = ""
        name_3 = ""
        while len(set([name_1, name_2, name_3])) < 3:
            name_1 = random.choice(names)
            name_2 = random.choice(names)
            name_3 = random.choice(names)

        nouns = {}
        ioi_prompt = {}
        for k in nouns_dict:
            nouns[k] = random.choice(nouns_dict[k])
            ioi_prompt[k] = nouns[k]
        prompt = temp
        for k in nouns_dict:
            prompt = prompt.replace(k, nouns[k])

        if prefixes is not None:
            L = random.randint(30, 40)
            pref = ".".join(random.choice(prefixes).split(".")[:L])
            pref += "<|endoftext|>"
        else:
            pref = ""

        prompt1 = prompt.replace("[A]", name_1)
        prompt1 = prompt1.replace("[B]", name_2)
        if abc:
            prompt1 = prompt1.replace("[C]", name_3)
        prompt1 = pref + prompt1
        ioi_prompt["text"] = prompt1
        ioi_prompt["IO"] = name_1
        ioi_prompt["S"] = name_2
        ioi_prompt["TEMPLATE_IDX"] = temp_id
        ioi_prompts.append(ioi_prompt)
        if abc:
            ioi_prompts[-1]["C"] = name_3

        nb_gen += 1

        if symmetric and nb_gen < N:
            prompt2 = prompt.replace("[A]", name_2)
            prompt2 = prompt2.replace("[B]", name_1)
            prompt2 = pref + prompt2
            ioi_prompts.append(
                {"text": prompt2, "IO": name_2, "S": name_1, "TEMPLATE_IDX": temp_id}
            )
            nb_gen += 1
    return ioi_prompts



def flip_words_in_prompt(prompt: str, word1: str, word2: str, instances: Optional[Union[int, List[int]]] = None):
    '''
    Flips instances of word `word1` with `word2` in the string `string`.

    By default it flips all instances, but the optional `instances` argument specifies which
    instances to flip (e.g. if instances = 0, then it only flips the 0th instance of either
    word1 or word2.

    Examples of (arguments) -> return value:

        ("ABA", "A", "B") -> "BAB"
        ("ABA", "A", "B", 1) -> "AAA"
        ("ABA", "A", "B", [0, 1]) -> "BAA
    '''
    split_prompt = re.split("({}|{})".format(word1, word2), prompt)
    indices_of_names = [i for i, s in enumerate(split_prompt) if s in (word1, word2)]
    indices_to_flip = [indices_of_names[i] for i in instances]
    for i in indices_to_flip:
        split_prompt[i] = word1 if split_prompt[i] == word2 else word1
    prompt = "".join(split_prompt)
    return prompt
    


def gen_flipped_prompts(prompts: List[dict], templates_by_prompt: List[str], flip: str, names: List[str], seed: int) -> List[dict]:
    '''
    Flip prompts in a way described by the flip argument. Returns new prompts.

    prompts: List[dict]
        list of prompts, each prompt is a dict with keys "S", "IO", "text", etc

    templates_by_prompt: List[str]
        each element is "ABBA" or "BABA"

    flip: str
        "ABB -> XYZ, BAB -> XYZ" means that the prompt "A and B went to [place], B gave [object] to A" becomes "X and Y went to [place], Z gave [object] to A" (and equivalent for the BABA case)

    names: List[str]
        list of names, for when flip involves random tokens

    seed: int
        provides reproducibility

    Note that we don't bother flipping the last token in the prompt (IO2), since
    we don't use it for anything (intuitively, we use this function to create 
    datasets to provide us with corrupted signals, but we still use the IO2 from
    the original uncorrupted IOI database as our "correct answer", so we don't 
    care about what the correct answer (IO2) for the corrupted set is).
    '''
    random.seed(seed)
    np.random.seed(seed)
    abba_flip, baba_flip = flip.split(",")
    flip_dict = {
        "ABB": [flip.strip() for flip in abba_flip.split("->")],
        "BAB": [flip.strip() for flip in baba_flip.split("->")]
    }

    new_prompts = []
    
    for idx, (prompt, template) in enumerate(zip(prompts, templates_by_prompt)):

        flip_orig, flip_new = flip_dict[template[:-1]]

        prompt = copy.copy(prompt)

        # Get indices and original values of first three names int the prompt
        prompt_split = prompt["text"].split(" ")
        orig_names_and_posns = [(i, s) for i, s in enumerate(prompt_split) if s in names][:3]
        orig_names = list(zip(*orig_names_and_posns))[1]

        # Get a dictionary of the correspondence between orig names and letters in flip_orig
        # (and get a subdict for those names which are kept in flip_new)
        orig_names_key = {
            letter: s
            for s, letter in zip(orig_names, flip_orig)
        }
        kept_names_key = {
            k: v
            for k, v in orig_names_key.items() if k in flip_new
        }
        # This line will throw an error if flip_orig is wrong (e.g. if it says "SOS" but the
        # S1 and S2 tokens don't actually match
        assert len(orig_names_key) == len(set(flip_orig))
        
        # Get all random names we'll need, in the form of a dictionary
        rand_names = {
            letter: np.random.choice(list(set(names) - set(orig_names)))
            for letter in set(flip_new) - set(flip_orig)
        }
        
        # Get a "full dictionary" which maps letters in flip_new to the new values they will have
        name_replacement_dict = {**kept_names_key, **rand_names}
        assert len(name_replacement_dict) == len(set(flip_new)), (name_replacement_dict, flip_new)

        # Populate the new names, with either random names or with the corresponding orig names
        for (i, s), letter in zip(orig_names_and_posns, flip_new):
            prompt_split[i] = name_replacement_dict[letter]

        # Join the prompt back together
        prompt["text"] = " ".join(prompt_split)

        # Change the identity of the S and IO tokens.
        # S token is just same as S2, but IO is a bit messier because it might not be 
        # well-defined (it's defined as the unique non-duplicated name of the first 
        # two). If it's ill-defined, WLOG set it to be the second name.
        prompt["S"] = name_replacement_dict[flip_new[-1]]
        possible_IOs = [name_replacement_dict[letter] for letter in flip_new[:2] if list(flip_new).count(letter) == 1]
        # Case where IO is well-defined
        if len(possible_IOs) == 1:
            prompt["IO"] = possible_IOs[0]
        # Case where it isn't well-defined
        else:
            prompt["IO"] = name_replacement_dict[flip_new[1]]

        new_prompts.append(prompt)

    return new_prompts



def get_name_idxs(prompts, tokenizer, idx_types=["IO", "S1", "S2"], prepend_bos=False):
    name_idx_dict = dict((idx_type, []) for idx_type in idx_types)
    for prompt in prompts:
        text_split = prompt["text"].split(" ")
        toks = tokenizer.tokenize(" ".join(text_split[:-1]))
        # Get the first instance of IO token
        name_idx_dict["IO"].append(
            toks.index(tokenizer.tokenize(" " + prompt["IO"])[0])
        )
        # Get the first instance of S token
        try:
            name_idx_dict["S1"].append(
                toks.index(tokenizer.tokenize(" " + prompt["S"])[0]))
        except:
            print(prompt)
            print(tokenizer.tokenize(" " + prompt["S"]))
            print(toks)
            print(" ".join(text_split[:-1]))
            raise Exception("S1 token not found")
        # Get the last instance of S token
        name_idx_dict["S2"].append(
            len(toks) - toks[::-1].index(tokenizer.tokenize(" " + prompt["S"])[0]) - 1
        )

    return [
        int(prepend_bos) + t.tensor(name_idx_dict[idx_type])
        for idx_type in idx_types
    ]


def get_word_idxs(prompts, word_list, tokenizer):
    """Get the index of the words in word_list in the prompts. Exactly one of the word_list word has to be present in each prompt"""
    idxs = []
    tokenized_words = [
        tokenizer.decode(tokenizer(word)["input_ids"][0]) for word in word_list
    ]
    for prompt in prompts:
        toks = [
            tokenizer.decode(t)
            for t in tokenizer(prompt["text"], return_tensors="pt", padding=True)[
                "input_ids"
            ][0]
        ]
        idx = None
        for i, w_tok in enumerate(tokenized_words):
            if word_list[i] in prompt["text"]:
                try:
                    idx = toks.index(w_tok)
                    if toks.count(w_tok) > 1:
                        idx = len(toks) - toks[::-1].index(w_tok) - 1
                except:
                    idx = toks.index(w_tok)
                    # raise ValueError(toks, w_tok, prompt["text"])
        if idx is None:
            raise ValueError(f"Word {word_list} and {i} not found {prompt}")
        idxs.append(idx)
    return t.tensor(idxs)


def get_end_idxs(toks, tokenizer, name_tok_len=1, prepend_bos=False):
    relevant_idx = int(prepend_bos)
    # if the sentence begins with an end token
    # AND the model pads at the end with the same end token,
    # then we need make special arrangements

    pad_token_id = tokenizer.pad_token_id

    end_idxs_raw = []
    for i in range(toks.shape[0]):
        if pad_token_id not in toks[i][1:]:
            end_idxs_raw.append(toks.shape[1])
            continue
        nonzers = (toks[i] == pad_token_id).nonzero()[relevant_idx][0].item()
        end_idxs_raw.append(nonzers)
    end_idxs = t.tensor(end_idxs_raw)
    end_idxs = end_idxs - 1 - name_tok_len

    for i in range(toks.shape[0]):
        assert toks[i][end_idxs[i] + 1] != 0 and (
            toks.shape[1] == end_idxs[i] + 2 or toks[i][end_idxs[i] + 2] == pad_token_id
        ), (
            toks[i],
            end_idxs[i],
            toks[i].shape,
            "the END idxs aren't properly formatted",
        )

    return end_idxs



def get_idx_dict(ioi_prompts, tokenizer, prepend_bos=False, toks=None):
    (IO_idxs, S1_idxs, S2_idxs,) = get_name_idxs(
        ioi_prompts,
        tokenizer,
        idx_types=["IO", "S1", "S2"],
        prepend_bos=prepend_bos,
    )

    end_idxs = get_end_idxs(
        toks,
        tokenizer,
        name_tok_len=1,
        prepend_bos=prepend_bos,
    )

    punct_idxs = get_word_idxs(ioi_prompts, [",", "."], tokenizer)

    return {
        "IO": IO_idxs,
        "IO-1": IO_idxs - 1,
        "IO+1": IO_idxs + 1,
        "S1": S1_idxs,
        "S1-1": S1_idxs - 1,
        "S1+1": S1_idxs + 1,
        "S2": S2_idxs,
        "end": end_idxs,
        "starts": t.zeros_like(end_idxs),
        "punct": punct_idxs,
    }

PROMPT_DETAIL = {
    "ABBA": {'NORMAL': ABBA_TEMPLATES, 'LONG': ABBA_LONG_TEMPLATES, 
             'EXPANDED': ABBA_EXPANDED_TEMPLATES, 'COMPLEMENT': ABBA_COMPLEMENT_TEMPLATES,
             'EXTRA_AND': ABBA_EXTRA_AND_TEMPLATES, 'NAME_A': ABBA_NAME_A_PREFIX_TEMPLATES,
             'NAME_B': ABBA_NAME_B_PREFIX_TEMPLATES, 'EXTRA_AND2': ABBA_EXTRA_AND2_TEMPLATES},
    "BABA": {'NORMAL': BABA_TEMPLATES, 'LONG': BABA_LONG_TEMPLATES,
             'EXPANDED': BABA_EXPANDED_TEMPLATES, 'COMPLEMENT': BABA_COMPLEMENT_TEMPLATES,
             'EXTRA_AND': BABA_EXTRA_AND_TEMPLATES, 'NAME_A': BABA_NAME_A_PREFIX_TEMPLATES,
             'NAME_B': BABA_NAME_B_PREFIX_TEMPLATES, 'EXTRA_AND2': BABA_EXTRA_AND2_TEMPLATES},
}

class IOIDataset:
    def __init__(
        self,
        prompt_type: Union[
            str, List[str]
        ],  # if list, then it will be a list of templates
        N=500,
        tokenizer=None,
        prompts=None,
        symmetric=False,
        prefixes=None,
        nb_templates=None,
        prepend_bos=False,
        manual_word_idx=None,
        has_been_flipped:bool=False,
        prompt_detail='NORMAL',
        toks_len:Optional[int]=None,
        seed=0,
        device="cuda"
    ):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        if not (
            N == 1
            or prepend_bos == False
            or tokenizer.bos_token_id == tokenizer.eos_token_id
        ):
            warnings.warn(
                "Probably word_idx will be calculated incorrectly due to this formatting"
            )
        self.has_been_flipped = has_been_flipped
        assert not (symmetric and prompt_type == "ABC")
        assert (
            (prompts is not None) or (not symmetric) or (N % 2 == 0)
        ), f"{symmetric} {N}"
        self.prompt_type = prompt_type

        if nb_templates is None:
            nb_templates = len(BABA_TEMPLATES)

        abba_templates, baba_templates = PROMPT_DETAIL['ABBA'][prompt_detail], PROMPT_DETAIL['BABA'][prompt_detail]

        if prompt_type == "ABBA":
            self.templates = abba_templates[:nb_templates].copy()
        elif prompt_type == "BABA":
            self.templates = baba_templates[:nb_templates].copy()
        elif prompt_type == "mixed":
            self.templates = (
                baba_templates[: nb_templates // 2].copy()
                + abba_templates[: nb_templates // 2].copy()
            )
            random.shuffle(self.templates)
        
        elif prompt_type == "ABC":
            self.templates = ABC_TEMPLATES[:nb_templates].copy()
        elif prompt_type == "BAC":
            self.templates = BAC_TEMPLATES[:nb_templates].copy()
        elif prompt_type == "ABC mixed":
            self.templates = (
                ABC_TEMPLATES[: nb_templates // 2].copy()
                + BAC_TEMPLATES[: nb_templates // 2].copy()
            )
            random.shuffle(self.templates)
        elif isinstance(prompt_type, list):
            self.templates = prompt_type
        else:
            raise ValueError(prompt_type)

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        self.prefixes = prefixes
        self.prompt_type = prompt_type
        if prompts is None:
            self.ioi_prompts = gen_prompt_uniform(  # list of dict of the form {"text": "Alice and Bob bla bla. Bob gave bla to Alice", "IO": "Alice", "S": "Bob"}
                self.templates,
                NAMES,
                nouns_dict={"[PLACE]": PLACES, "[OBJECT]": OBJECTS},
                N=N,
                symmetric=symmetric,
                prefixes=self.prefixes,
                abc=(prompt_type in ["ABC", "ABC mixed", "BAC"]),
            )
        else:
            assert N == len(prompts), f"{N} and {len(prompts)}"
            self.ioi_prompts = prompts

        all_ids = [prompt["TEMPLATE_IDX"] for prompt in self.ioi_prompts]
        all_ids_ar = np.array(all_ids)
        self.groups = []
        for id in list(set(all_ids)):
            self.groups.append(np.where(all_ids_ar == id)[0])

        small_groups = []
        for group in self.groups:
            if len(group) < 5:
                small_groups.append(len(group))

        self.sentences = [
            prompt["text"] for prompt in self.ioi_prompts
        ]  # a list of strings. Renamed as this should NOT be forward passed

        self.templates_by_prompt = []  # for each prompt if it's ABBA or BABA
        for i in range(N):
            if self.sentences[i].index(self.ioi_prompts[i]["IO"]) < self.sentences[
                i
            ].index(self.ioi_prompts[i]["S"]):
                self.templates_by_prompt.append("ABBA")
            else:
                self.templates_by_prompt.append("BABA")

        texts = [
            (self.tokenizer.bos_token if prepend_bos else "") + prompt["text"]
            for prompt in self.ioi_prompts
        ]

        self.toks_len = toks_len
        toks = t.Tensor(self.tokenizer(texts, padding=True).input_ids).long()
        if toks_len is None:
            self.toks = toks
        else:
            self.toks = t.nn.functional.pad(toks, (0, toks_len - toks.shape[1]), value=tokenizer.eos_token_id)


        self.word_idx = get_idx_dict(
            self.ioi_prompts,
            self.tokenizer,
            prepend_bos=prepend_bos,
            toks=self.toks,
        )
        self.prepend_bos = prepend_bos
        if manual_word_idx is not None:
            self.word_idx = manual_word_idx

        self.N = N
        self.max_len = max(
            [
                len(self.tokenizer(prompt["text"]).input_ids)
                for prompt in self.ioi_prompts
            ]
        )

        self.io_tokenIDs = [
            self.tokenizer.encode(" " + prompt["IO"])[0] for prompt in self.ioi_prompts
        ]
        self.s_tokenIDs = [
            self.tokenizer.encode(" " + prompt["S"])[0] for prompt in self.ioi_prompts
        ]

        self.tokenized_prompts = []

        for i in range(self.N):
            self.tokenized_prompts.append(
                "|".join([self.tokenizer.decode(tok) for tok in self.toks[i]])
            )

        self.device = device
        self.to(device)
    
    def gen_flipped_prompts(self, flip):
        # Check if it's already been flipped (shouldn't string 2 flips together)
        if self.has_been_flipped:
            warnings.warn("This dataset has already been flipped. Generally, you should try and apply flips in one step, because this can lead to errors.")

        # Redefine seed (so it's different depending on what the flip is, e.g. we don't want (IO, RAND) then (S, RAND) to give us the same rand names)
        seed = self.seed + sum(map(ord, list("".join(flip))))

        # Get flipped prompts
        flipped_prompts = gen_flipped_prompts(self.ioi_prompts, self.templates_by_prompt, flip, NAMES, seed)

        flipped_ioi_dataset = IOIDataset(
            prompt_type=self.prompt_type,
            N=self.N,
            tokenizer=self.tokenizer,
            prompts=flipped_prompts,
            prefixes=self.prefixes,
            prepend_bos=self.prepend_bos,
            # manual_word_idx=self.word_idx,
            has_been_flipped=True,
            toks_len=self.toks_len,
            seed=seed
        )
        return flipped_ioi_dataset

    def copy(self):
        copy_ioi_dataset = IOIDataset(
            prompt_type=self.prompt_type,
            N=self.N,
            tokenizer=self.tokenizer,
            prompts=self.ioi_prompts.copy(),
            prefixes=self.prefixes.copy() if self.prefixes is not None else self.prefixes,
        )
        return copy_ioi_dataset

    def __getitem__(self, key):
        sliced_prompts = self.ioi_prompts[key]
        sliced_dataset = IOIDataset(
            prompt_type=self.prompt_type,
            N=len(sliced_prompts),
            tokenizer=self.tokenizer,
            prompts=sliced_prompts,
            prefixes=self.prefixes,
            prepend_bos=self.prepend_bos,
        )
        return sliced_dataset

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __delitem__(self, key):
        raise NotImplementedError()

    def __len__(self):
        return self.N

    def tokenized_prompts(self):
        return self.toks

    def to(self, device):
        self.toks = self.toks.to(device)
        return self
