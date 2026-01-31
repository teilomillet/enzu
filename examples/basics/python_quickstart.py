from enzu import Enzu, ask

print(ask("What is 2+2?"))

client = Enzu()
answer = client.run(
    "Summarize the key points",
    data="...long document...",
    tokens=400,
)
print(answer)
