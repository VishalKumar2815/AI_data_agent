import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from utils.insights_tool import InsightsTools
from db_loader import get_conn_details
from Models.schema import InsightsAgentSchema
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END


def gather_summary(state: InsightsAgentSchema) -> InsightsAgentSchema:
    tools = InsightsTools(get_conn_details())
    state.data_summary = tools.build_data_summary(state.schema_name, state.table_name or None)
    return state


def generate_insights(state: InsightsAgentSchema) -> InsightsAgentSchema:
    llm = pick_llm("low")

    prompt = f"""
    You are a data analyst. Based on the statistical summary of the dataset(s)
    below, answer the user's question with clear, concise, business-friendly
    key insights — notable trends, distributions, data quality issues (nulls,
    outliers), and anything else genuinely useful. Do not include raw code or
    SQL. Keep it readable, use bullet points where it helps.

    User's question: {state.user_question}

    Data summary:
    {state.data_summary}
    """

    response = llm.invoke(prompt).content

    state.final_answer = response
    state.messages = state.messages + [AIMessage(content=response)]

    return state


insights_graph = StateGraph(InsightsAgentSchema)
insights_graph.add_node("gather_summary", gather_summary)
insights_graph.add_node("generate_insights", generate_insights)

insights_graph.add_edge(START, "gather_summary")
insights_graph.add_edge("gather_summary", "generate_insights")
insights_graph.add_edge("generate_insights", END)

insights_analyst = insights_graph.compile()


if __name__ == "__main__":
    response = insights_analyst.invoke(
        {
            "messages": [],
            "user_question": "What are the key insights from this data?",
            "schema_name": "public",
            "table_name": "",
            "data_summary": "",
            "final_answer": "",
        }
    )
    print(response["final_answer"])