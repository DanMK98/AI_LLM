import torch
import torch.nn as nn
from tokenizer import characters, tokens, id_to_char

torch.manual_seed(42)

# Training examples: current char -> next char
inputs = torch.tensor(tokens[:-1], dtype=torch.long) 
targets = torch.tensor(tokens[1:], dtype=torch.long) 

vocab_size = len(characters)

# Each character gets a row of scores for possible next characters
model = nn.Embedding(vocab_size, vocab_size)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(300):
    scores = model(inputs) 
    loss = loss_function(scores, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

# Inspect the learned predictions
with torch.no_grad():
    for i in range(vocab_size):
        scores = model(torch.tensor([i]))
        predicted_id = scores.argmax(dim=1).item()

        print(f"{id_to_char[i]!r} → {id_to_char[predicted_id]!r}")

# Start with the first character of the training text and generate 50 more characters
current_id = tokens[0]
generated = [current_id]

with torch.no_grad():
    for _ in range(50):
        scores = model(torch.tensor([current_id]))

        # Convert scores to probabilities
        probabilities = torch.softmax(scores, dim=1)

        # Randomly sample the next character based on probabilities
        current_id = torch.multinomial(probabilities[0], num_samples=1).item()

        generated.append(current_id)

result = ''.join(id_to_char[i] for i in generated)
print("Generated text:", result)
    