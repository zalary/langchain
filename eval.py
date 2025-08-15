from langsmith.client import Client
from langchain_groq import ChatGroq
from judge import HelpfulnessJudge
from langsmith.evaluation import RunEvaluator, EvaluationResult
from langchain.evaluation import load_evaluator
from langsmith.run_helpers import trace
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

# Manual test case evaluation
def evaluate_test_cases():
    evaluator = TierAccessJudge()
    for case in test_cases:
        if "expect_access" not in case:
            raise ValueError(f"Missing 'expect_access' in test case: {case['name']}")
        inputs = case["inputs"]
        expected = case["expect_access"]

        output = graph.invoke(inputs, config={"configurable": {"user_name": "LangSmith"}})

        run = client.create_run(
            run_type="chain",
            name=case.get("name", "Unnamed Case"),
            inputs=inputs,
            outputs=output,
            project_name=PROJECT_NAME,
            reference_output="should allow access" if expected else "should deny access",
        )

        if run is None or not hasattr(run, "inputs") or run.inputs is None:
            print(f"Skipping case {case.get('name', 'Unnamed Case')} due to missing run inputs")
            continue

        result: EvaluationResult = evaluator.evaluate_run(run, {"expect_access": expected})
        client.log_evaluation(run.id, result)
        print(f"{case['name']}: {result.score} — {result.comment}")

if __name__ == "__main__":
    # Choose one:
    # evaluate_with_dataset("Support Questions")
    evaluate_test_cases()