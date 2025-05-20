from rag_no_img import RAGAgent  # Replace with actual filename (no .py)
agent = RAGAgent(data_dir="./Dataset")

# Step 1: Ingest PDFs
agent.indexPDF()

# Step 2: Ask a question
query = "What was the revenue Apple in 2023?"
results = agent.generate_answer(query, top_k=3)
print("\n📝 Text Results:")
print(results["answer"])  # Print the model's actual answer

print("\n📄 Source Documents:")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"- {doc[:200]}...\n  ↪ Metadata: {meta}")

# # Step 3: Print results
# print("\n📝 Text Results:")
# for doc, meta in zip(results["documents"][0], results["metadatas"][0]):

#     print(f"- {doc[:200]}...\n  ↪ Metadata: {meta}")







