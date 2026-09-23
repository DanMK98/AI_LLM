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