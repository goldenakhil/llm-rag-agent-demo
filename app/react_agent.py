from typing import List, Dict, Any
from openai import OpenAI

from .rag import RagPipeline

CHAT_MODEL = "gpt-4o-mini"  # adjust if needed


class ReactAgent:
    """
    Minimal ReAct-style agent:
    - Receives a user query
    - Can choose to Search[query] using RAG
    - Or Answer[final answer]
    - Runs for a small fixed number of steps
    """

    def __init__(self, client: OpenAI, rag: RagPipeline, max_steps: int = 3):
        self.client = client
        self.rag = rag
        self.max_steps = max_steps

    def _system_prompt(self) -> str:
        return (
            "You are a helpful assistant that uses a ReAct pattern.\n"
            "You have access to one tool:\n\n"
            "Tool: Search[query]\n"
            " - Use this when you need to look up information in the knowledge base.\n"
            " - It returns relevant document chunks.\n\n"
            "When reasoning, follow this format:\n"
            "Thought: ...\n"
            "Action: Search[query]\n"
            "or\n"
            "Thought: ...\n"
            "Action: Answer[final answer here]\n"
        )

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content

    def run(self, query: str) -> Dict[str, Any]:
        """
        Returns:
            {
                "answer": str,
                "steps": [ {thought, action, observation}, ... ],
                "retrieved_chunks": [ {id, filename, text}, ... ]
            }
        """
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": f"User question: {query}"},
        ]

        steps: List[Dict[str, Any]] = []
        retrieved_chunks: List[Dict[str, str]] = []

        for _ in range(self.max_steps):
            llm_output = self._call_llm(messages)

            thought_line = None
            action_line = None
            for line in llm_output.splitlines():
                if line.startswith("Thought:"):
                    thought_line = line[len("Thought:"):].strip()
                if line.startswith("Action:"):
                    action_line = line[len("Action:"):].strip()

            if not action_line:
                # Fallback: treat entire output as answer
                steps.append(
                    {"thought": thought_line or "", "action": "Answer", "observation": ""}
                )
                return {
                    "answer": llm_output,
                    "steps": steps,
                    "retrieved_chunks": retrieved_chunks,
                }

            # Handle Search[...] and Answer[...]
            if action_line.startswith("Search[") and action_line.endswith("]"):
                search_query = action_line[len("Search["):-1].strip() or query

                context_str, docs = self.rag.build_context(search_query)
                observation = f"Retrieved documents:\n{context_str}"

                messages.append({"role": "assistant", "content": llm_output})
                messages.append({"role": "user", "content": f"Observation: {observation}"})

                steps.append(
                    {
                        "thought": thought_line or "",
                        "action": f"Search[{search_query}]",
                        "observation": "Retrieved documents from RAG.",
                    }
                )

                for d in docs:
                    retrieved_chunks.append(
                        {
                            "id": d.id,
                            "filename": d.metadata.get("filename", d.id),
                            "text": d.text,
                        }
                    )

            elif action_line.startswith("Answer[") and action_line.endswith("]"):
                final_answer = action_line[len("Answer["):-1].strip()
                steps.append(
                    {
                        "thought": thought_line or "",
                        "action": "Answer",
                        "observation": "",
                    }
                )
                return {
                    "answer": final_answer,
                    "steps": steps,
                    "retrieved_chunks": retrieved_chunks,
                }
            else:
                # Unknown format → treat as final answer
                steps.append(
                    {"thought": thought_line or "", "action": "Answer", "observation": ""}
                )
                return {
                    "answer": llm_output,
                    "steps": steps,
                    "retrieved_chunks": retrieved_chunks,
                }

        # Ran out of steps
        return {
            "answer": "I reached the maximum number of reasoning steps without a final answer.",
            "steps": steps,
            "retrieved_chunks": retrieved_chunks,
        }
