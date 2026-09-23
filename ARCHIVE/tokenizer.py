from pathlib import Path

text_path = Path(__file__).parent / "input.txt"
text = text_path.read_text(encoding="utf-8")
assert text, "Input text is empty. Please provide a valid input.txt file."

# Collect unique characters in alphabetical order
characters = sorted(set(text))

# Map characters to numbers, and numbers back to characters
char_to_id = {char: i for i, char in enumerate(characters)}
id_to_char = {i: char for i, char in enumerate(characters)}

# Encode text into numbers
tokens = [char_to_id[char] for char in text]

# Decode numbers back into text
decoded = ''.join(id_to_char[token] for token in tokens)

print("Total characters:", len(text))
print("Vocabulary size:", len(characters))
print("First 20 characters:", tokens[:20])

assert decoded == text, "Decoded text does not match original text"

# All tokens except the last one
inputs = tokens[:-1]

# All tokens except the first one
targets = tokens[1:]