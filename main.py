from graph import build_graph

if __name__ == "__main__":
    app = build_graph()

    result = app.invoke({
        "query": "A person cheated another by forging documents to sell land."
    })

    print("\nFINAL LEGAL OPINION:\n")
    print(result["final_opinion"])
