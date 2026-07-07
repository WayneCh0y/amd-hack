"""Labelled benchmark tasks: 3 per category across all 8 capability areas.

Each task pairs a prompt with a deterministic ``check`` (see ``checkers.py``).
The prompts are original and phrased plainly — including a few that would fool
keyword routing — so results reflect genuine capability, not memorised answers.
Extend this set freely; more tasks = more reliable per-category signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.categories import Category


@dataclass(frozen=True)
class BenchTask:
    id: str
    category: Category
    prompt: str
    check: dict = field(default_factory=dict)


TASKS: list[BenchTask] = [
    # --- Factual knowledge -------------------------------------------------
    BenchTask("fact1", Category.FACTUAL,
              "What is Newton's First Law of Motion?",
              {"type": "keywords",
               "any": [["rest"], ["motion", "moving", "uniform"],
                       ["force", "unbalanced", "external", "unless"]]}),
    BenchTask("fact2", Category.FACTUAL,
              "At sea level, what is the boiling point of water in degrees Celsius?",
              {"type": "numeric", "value": 100, "tol": 0.5}),
    BenchTask("fact3", Category.FACTUAL,
              "What is the capital city of France?",
              {"type": "keywords", "all": ["paris"]}),

    # --- Mathematical reasoning -------------------------------------------
    BenchTask("math1", Category.MATH,
              "A shirt costs $40 and is discounted by 25%. An 8% sales tax is then "
              "applied to the discounted price. What is the final price?",
              {"type": "numeric", "value": 32.4, "tol": 0.02}),
    BenchTask("math2", Category.MATH,
              "What is 15% of 240?",
              {"type": "numeric", "value": 36, "tol": 0.01}),
    # Plainly-worded word problem with no math keywords (defeats keyword routing).
    BenchTask("math3", Category.MATH,
              "Tom is twice as old as Jerry. In five years, their combined age will "
              "be forty. How old is Jerry now?",
              {"type": "numeric", "value": 10, "tol": 0.01}),

    # --- Sentiment classification -----------------------------------------
    BenchTask("sent1", Category.SENTIMENT,
              "Classify the sentiment: 'I absolutely love this product, it works "
              "perfectly and exceeded my expectations!'",
              {"type": "label", "expected": ["positive"]}),
    BenchTask("sent2", Category.SENTIMENT,
              "Classify the sentiment: 'This is the worst purchase I have ever made "
              "and I want a refund.'",
              {"type": "label", "expected": ["negative"]}),
    BenchTask("sent3", Category.SENTIMENT,
              "Classify the sentiment: 'The battery life is fantastic, but the screen "
              "scratches far too easily.'",
              {"type": "label", "expected": ["neutral", "mixed"]}),

    # --- Text summarisation ------------------------------------------------
    BenchTask("sum1", Category.SUMMARIZATION,
              "Summarise in one sentence: Photosynthesis is the process by which green "
              "plants, algae and some bacteria convert light energy from the sun into "
              "chemical energy stored in glucose, releasing oxygen as a by-product.",
              {"type": "keywords",
               "any": [["light", "sun", "sunlight"],
                       ["glucose", "energy", "sugar"], ["oxygen"]]}),
    BenchTask("sum2", Category.SUMMARIZATION,
              "Summarise in one sentence: The water cycle describes how water "
              "evaporates from oceans and lakes, condenses into clouds, and returns to "
              "the surface as precipitation such as rain or snow.",
              {"type": "keywords",
               "any": [["water"], ["evapor", "condens", "precipitat", "cycle", "rain"]]}),
    BenchTask("sum3", Category.SUMMARIZATION,
              "Summarise in one sentence: The Great Barrier Reef, off the coast of "
              "Queensland, Australia, is the world's largest coral reef system and is "
              "home to thousands of species of marine life.",
              {"type": "keywords",
               "any": [["reef", "coral"], ["australia", "queensland"]]}),

    # --- Named entity recognition -----------------------------------------
    BenchTask("ner1", Category.NER,
              "Extract the named entities (person, organisation, location, date) from: "
              "'On 12 March 2023, Satya Nadella announced in Seattle that Microsoft "
              "would partner with OpenAI.'",
              {"type": "entities",
               "expected": ["Satya Nadella", "Seattle", "Microsoft", "OpenAI",
                            "March"]}),
    BenchTask("ner2", Category.NER,
              "Extract the named entities: 'Elon Musk founded SpaceX in California in "
              "2002.'",
              {"type": "entities",
               "expected": ["Elon Musk", "SpaceX", "California", "2002"]}),
    # Plainly-worded extraction request (no "entities"/"NER" keyword).
    BenchTask("ner3", Category.NER,
              "List the people, organisations and places mentioned here: 'Tim Cook "
              "spoke at Stanford University in Palo Alto.'",
              {"type": "entities",
               "expected": ["Tim Cook", "Stanford", "Palo Alto"]}),

    # --- Code debugging ----------------------------------------------------
    BenchTask("debug1", Category.CODE_DEBUG,
              "This function should return the factorial of n but has a bug. Provide "
              "the corrected function:\n```python\ndef factorial(n):\n    result = 0\n"
              "    for i in range(1, n + 1):\n        result *= i\n    return result\n```",
              {"type": "code",
               "tests": "assert factorial(5) == 120\nassert factorial(0) == 1\n"
                        "assert factorial(1) == 1"}),
    BenchTask("debug2", Category.CODE_DEBUG,
              "This function should return the sum of a list but returns the wrong "
              "value. Fix it:\n```python\ndef total(xs):\n    s = 0\n    for x in xs:\n"
              "        s = x\n    return s\n```",
              {"type": "code",
               "tests": "assert total([1,2,3]) == 6\nassert total([]) == 0\n"
                        "assert total([5]) == 5"}),
    BenchTask("debug3", Category.CODE_DEBUG,
              "This function should return the maximum value in a list but fails on "
              "all-negative lists. Fix it:\n```python\ndef max_val(xs):\n    m = 0\n"
              "    for x in xs:\n        if x > m:\n            m = x\n    return m\n```",
              {"type": "code",
               "tests": "assert max_val([-3,-1,-2]) == -1\nassert max_val([4,9,2]) == 9"}),

    # --- Logical / deductive reasoning ------------------------------------
    BenchTask("logic1", Category.LOGIC,
              "Ana, Ben and Cara each own a different pet: a cat, a dog and a fish. "
              "Ana does not own the dog. Ben owns the fish. Who owns the cat?",
              {"type": "keywords", "all": ["ana"]}),
    BenchTask("logic2", Category.LOGIC,
              "Alice, Bob and Carol finished a race in some order. Alice did not finish "
              "first. Carol finished last. Who finished first?",
              {"type": "keywords", "all": ["bob"]}),
    BenchTask("logic3", Category.LOGIC,
              "All Bloops are Razzies. All Razzies are Lazzies. Are all Bloops "
              "necessarily Lazzies? Answer yes or no.",
              {"type": "keywords", "any": [["yes"]]}),

    # --- Code generation ---------------------------------------------------
    BenchTask("gen1", Category.CODE_GEN,
              "Write a Python function `is_palindrome(s: str) -> bool` that returns "
              "True if the string is a palindrome, ignoring case, spaces and "
              "punctuation.",
              {"type": "code",
               "tests": "assert is_palindrome('A man, a plan, a canal: Panama')\n"
                        "assert not is_palindrome('hello')\nassert is_palindrome('')"}),
    BenchTask("gen2", Category.CODE_GEN,
              "Write a Python function `fib(n: int) -> int` that returns the nth "
              "Fibonacci number, 0-indexed, where fib(0)=0 and fib(1)=1.",
              {"type": "code",
               "tests": "assert fib(0) == 0\nassert fib(1) == 1\nassert fib(10) == 55"}),
    BenchTask("gen3", Category.CODE_GEN,
              "Write a Python function `count_vowels(s: str) -> int` that returns the "
              "number of vowels (a, e, i, o, u; case-insensitive) in the string.",
              {"type": "code",
               "tests": "assert count_vowels('Hello World') == 3\n"
                        "assert count_vowels('xyz') == 0\nassert count_vowels('AEIOU') == 5"}),
]
