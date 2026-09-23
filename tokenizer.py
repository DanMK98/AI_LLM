text = "hello world"

# Collect unique characters in alphabetical order
characters = sorted(set(text))

# Map characters to numbers, and numbers back to characters
char_to_id = {char: i for i, char in enumerate(characters)}
id_to_char = {i: char for i, char in enumerate(characters)}

# Encode text into numbers
tokens = [char_to_id[char] for char in text]

# Decode numbers back into text
decoded = ''.join(id_to_char[token] for token in tokens)

print("Characters:", characters)
print("Token IDs:", tokens)
print("Decoded Text:", decoded)

# All tokens except the last one
inputs = tokens[:-1]

# All tokens except the first one
targets = tokens[1:]

for current, following in zip(inputs, targets):
    print(f"{id_to_char[current]!r} → {id_to_char[following]!r}")