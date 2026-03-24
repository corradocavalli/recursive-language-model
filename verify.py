"""Ground truth verification for ai_history.txt questions."""

import re
from collections import defaultdict

with open("data/ai_history.txt") as f:
    text = f.read()

# (1) Distinct years - CE and BCE
bce_pattern = re.findall(r"(\d+)\s*BCE", text)
bce_years = set(bce_pattern)
all_4digit = set(re.findall(r"\b(\d{4})\b", text))
ce_years = {y for y in all_4digit if 1000 <= int(y) <= 2030}

print("=" * 60)
print("(1) DISTINCT YEARS")
print(f"    BCE years: {len(bce_years)} -> {sorted(bce_years)}")
print(f"    CE years:  {len(ce_years)}")
print(f"    CE list:   {sorted(ce_years)}")

# (2) Top 5 years by section/appendix count
# Use multiple section marker patterns
sections = re.split(
    r"\n(?=(?:Appendix [A-Z]|PART [IVX]+|Chapter \d|Section \d|"
    r"\d+\.\d+\s|---+\s*\n))",
    text,
)
print(f"\n{'=' * 60}")
print(f"(2) TOP 5 YEARS BY SECTION COUNT (total sections: {len(sections)})")
year_sections = defaultdict(set)
for i, sec in enumerate(sections):
    for y in set(re.findall(r"\b(\d{4})\b", sec)):
        if 1000 <= int(y) <= 2030:
            year_sections[y].add(i)
top5 = sorted(year_sections.items(), key=lambda x: len(x[1]), reverse=True)[:10]
for y, secs in top5:
    print(f"    {y}: {len(secs)} sections")

# (3) Years with BOTH hardware AND software/AI milestones
hw_kw = (
    r"hardware|GPU|CPU|chip|transistor|FLOPS|TPU|processor|circuit|"
    r"NVIDIA|AMD|Intel|computing|accelerator|DRAM|HBM|silicon|ASIC|"
    r"supercomputer|mainframe|microprocessor|semiconductor"
)
sw_kw = (
    r"AI\b|artificial intelligence|neural net|deep learning|"
    r"machine learning|GPT|BERT|AlphaGo|Watson|ELIZA|language model|"
    r"chatbot|NLP|computer vision|robot|expert system|perceptron|"
    r"reinforcement learning|generative|diffusion model|transformer"
)

hw_years = set()
sw_years = set()
for m in re.finditer(r"\b(\d{4})\b", text):
    y = m.group(1)
    if not (1800 <= int(y) <= 2030):
        continue
    start = max(0, m.start() - 500)
    end = min(len(text), m.end() + 500)
    ctx = text[start:end]
    if re.search(hw_kw, ctx, re.IGNORECASE):
        hw_years.add(y)
    if re.search(sw_kw, ctx, re.IGNORECASE):
        sw_years.add(y)

both = sorted(hw_years & sw_years, key=int)
print(f"\n{'=' * 60}")
print(f"(3) YEARS WITH BOTH HW + SW MILESTONES: {len(both)}")
print(f"    {both}")

# (4) Longest gap 1900-2026
mentioned = sorted({int(y) for y in ce_years if 1900 <= int(y) <= 2026})
print(f"\n{'=' * 60}")
print(f"(4) LONGEST GAP (1900-2026)")
print(f"    Years mentioned in range: {len(mentioned)}")
print(f"    First: {mentioned[0]}, Last: {mentioned[-1]}")

max_gap = 0
gap_info = ""
# Check gap from 1900 to first mentioned year
first_gap = mentioned[0] - 1900
if first_gap > max_gap:
    max_gap = first_gap
    gap_info = f"1900-{mentioned[0] - 1}"
# Check gaps between consecutive mentioned years
for i in range(1, len(mentioned)):
    gap = mentioned[i] - mentioned[i - 1] - 1
    if gap > max_gap:
        max_gap = gap
        gap_info = f"{mentioned[i - 1] + 1}-{mentioned[i] - 1}"
# Check trailing gap
trailing = 2026 - mentioned[-1]
if trailing > max_gap:
    max_gap = trailing
    gap_info = f"{mentioned[-1] + 1}-2026"

print(f"    Longest gap: {max_gap} years ({gap_info})")

# (5) Named AI systems
print(f"\n{'=' * 60}")
print("(5) NAMED AI SYSTEMS/PROJECTS")
# Broad pattern: look for capitalized multi-word names or acronyms near AI context
ai_names = set()
# Known AI systems mentioned in any AI history document
known_patterns = [
    r"Logic Theorist",
    r"General Problem Solver",
    r"GPS\b",
    r"ELIZA",
    r"SHRDLU",
    r"MYCIN",
    r"DENDRAL",
    r"PROSPECTOR",
    r"Deep Blue",
    r"Watson\b",
    r"AlphaGo",
    r"AlphaZero",
    r"AlphaFold",
    r"AlphaStar",
    r"AlphaTensor",
    r"AlphaProof",
    r"AlphaCode",
    r"GPT-[1-5]\w*",
    r"BERT\b",
    r"DALL[-·]E\s?\d?",
    r"ChatGPT",
    r"Codex\b",
    r"GitHub Copilot",
    r"Copilot\b",
    r"Stable Diffusion",
    r"Midjourney",
    r"MuZero",
    r"DQN\b",
    r"LeNet",
    r"AlexNet",
    r"GoogLeNet",
    r"VGGNet",
    r"ResNet\b",
    r"Transformer\b",
    r"SAINT\b",
    r"STUDENT\b",
    r"ANALOGY\b",
    r"Sora\b",
    r"Gemini\b",
    r"Claude\b",
    r"LLaMA\b",
    r"Mistral\b",
    r"Perceptron\b",
    r"TD-Gammon",
    r"Stanley\b",
    r"DeepSeek",
    r"Jukebox",
    r"MuseNet",
    r"MusicLM",
    r"Khanmigo",
    r"R1/XCON",
    r"XCON\b",
    r"Shakey",
    r"SNARC\b",
    r"Chinchilla",
    r"PaLM\b",
    r"PaLM-E",
    r"Bard\b",
    r"BLOOM\b",
    r"Falcon\b",
    r"Megatron",
    r"Turing-NLG",
    r"CLIP\b",
    r"Whisper\b",
    r"Segment Anything",
    r"ImageNet\b",
    r"Word2Vec",
    r"GloVe\b",
    r"ELMo\b",
    r"T5\b",
    r"XLNet",
    r"RoBERTa",
    r"DistilBERT",
    r"Attention\b.*mechanism",
    r"seq2seq",
    r"GAN\b",
    r"VAE\b",
    r"LSTM\b",
    r"RNN\b",
    r"CNN\b",
    r"Siri\b",
    r"Alexa\b",
    r"Cortana\b",
    r"TensorFlow",
    r"PyTorch",
    r"Keras\b",
    r"Theano\b",
    r"Caffe\b",
    r"OpenAI\b",
    r"DeepMind",
    r"Anthropic",
    r"LISP\b",
    r"Prolog\b",
    r"STRIPS\b",
    r"Neuralink",
    r"Tesla.*Autopilot",
    r"WaveNet",
    r"StyleGAN",
    r"BigGAN",
    r"ProGAN",
    r"Neural.*Turing.*Machine",
    r"Memory.*Network",
]
for pat in known_patterns:
    matches = re.findall(pat, text, re.IGNORECASE)
    for m in matches:
        ai_names.add(m.strip())

print(f"    Found (known patterns): {len(ai_names)}")
# Print them sorted
for name in sorted(ai_names):
    print(f"      - {name}")
