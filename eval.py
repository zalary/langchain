from langsmith.client import Client
from langchain_groq import ChatGroq
from judge import HelpfulnessJudge
from langsmith.evaluation import RunEvaluator, EvaluationResult
from langchain.evaluation import load_evaluator
from langsmith.run_helpers import trace
from langsmith import traceable
from agent import graph  # your compiled LangGraph
from dataset import test_cases
from dotenv import load_dotenv
load_dotenv()

PROJECT_NAME = "langgraph-tiered-support"
client = Client()

# Optional: Evaluate using a dataset from LangSmith
def evaluate_with_dataset(dataset_name: str):
    dataset = client.read_dataset(name=dataset_name)

    client.run_on_dataset(
        dataset_name=dataset.name,
        llm_or_chain_factory=lambda: graph,
        evaluators=[HelpfulnessJudge()],
        input_mapper=lambda x: x,
        verbose=True,
    )

# Custom evaluator: Did the assistant enforce access correctly?
class TierAccessJudge(RunEvaluator):
    def __init__(self):
        groq_llm = ChatGroq(temperature=0, model="llama3-70b-8192")
        self.judge = load_evaluator(
            "labeled_criteria",
            criteria={"correctness": "Was the access decision correct?"},
            llm=groq_llm 
        )

    def evaluate_run(self, run, example=None):
        question = run.inputs["messages"][-1]["content"]
        answer = run.outputs["messages"][-1]["content"]
        expected = example["expect_access"]
        reference = "should allow access" if expected else "should deny access"

        return self.judge.evaluate_strings(
            input=question,
            prediction=answer,
            reference=reference,
        )
    
@traceable(name="eval_case")
def run_case(inputs: dict, tier: str, user_name: str, case_name: str, expect_access: bool):
    # inner span so you get tags/metadata on the invoke
    with trace(
        "graph.invoke",
        tags=[f"tier:{tier}"],
        metadata={"case": case_name, "expect_access": expect_access},
    ):
        return graph.invoke(
            inputs,
            config={"configurable": {"user_name": user_name, "customer_tier": tier}},
        )
    
# Manual test case evaluation
def evaluate_test_cases():
    evaluator = TierAccessJudge()

    for case in test_cases:
        name = case.get("name", "Unnamed Case")
        # Guard: must have expected label
        if "expect_access" not in case:
            print(f"Skipping {name}: missing 'expect_access'")
            continue
        expected = case["expect_access"]
        inputs = {"messages": case["inputs"]["messages"]}

        # pull tier from the case's inputs for the config
        tier = case.get("customer_tier") or case["inputs"].get("customer_tier", "Free")
        user_name = case.get("user_name", "LangSmith")

        print(f"[EVAL] {case.get('name', 'Unnamed')} → tier={tier}")

        # Invoke the graph with config carrying immutable context (tier, user name)
        try:
            output = run_case(inputs, tier, user_name, name, expected)
        except Exception as e:
            print(f"[EVAL] {name} → graph error: {e}")
            continue
            
        # Log the run to LangSmith
        run = client.create_run(
            run_type="chain",
            name=name,
            inputs=inputs,
            outputs=output,
            project_name=PROJECT_NAME,
            reference_output="should allow access" if expected else "should deny access",
        )

        # Safety guard: ensure run has inputs before evaluating
        if not getattr(run, "inputs", None):
            print(f"Skipping {name}: missing run inputs")
            continue

        # Evaluate with the custom judge
        result: EvaluationResult = evaluator.evaluate_run(run, {"expect_access": expected})
        client.log_evaluation(run.id, result)
        print(f"{name}: {getattr(result, 'score', None)} — {getattr(result, 'comment', '')}")

    print("[EVAL] done.")

if __name__ == "__main__":
    # Choose one:
    # evaluate_with_dataset("Support Questions")
    evaluate_test_cases()