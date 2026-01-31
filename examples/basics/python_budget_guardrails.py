from enzu import Enzu, Limits

client = Enzu()

result = client.run(
    "Summarize the risks and mitigations in one paragraph.",
    data="...long document...",
    limits=Limits(tokens=300, seconds=20, cost_usd=0.02),
)
print(result)
