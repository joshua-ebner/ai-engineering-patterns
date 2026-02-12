from langchain_core.messages import HumanMessage
from .agent_graph import agent



def main():
    print("\nAgent ready. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in ["exit", "quit"]:
            break

        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]}
        )

        print(
            "\nAssistant:",
            result["messages"][-1].content,
            "\n"
        )


if __name__ == "__main__":
    main()
