from langsmith.client import Client
from langsmith.evaluation import RunEvaluator, EvaluationResult
from langchain.chat_models import ChatOpenAI
from langchain.evaluation import load_evaluator
from agent import graph  # your compiled LangGraph
from test_cases import test_cases

PROJECT_NAME = "langgraph-tiered-support"

# Custom evaluator: did the assistant enforce access policy correctly?
class TierAccessJudge(RunEvaluator):
    def __init__(self):
        self.judge = load_evaluator(
            "labeled_criteria",
            criteria=["correctness"],
            llm=ChatOpenAI(temperature=0)
        )

    def evaluate_run(self, run, example=None):
        # Pass input, output, and expected access to judge
        user_input = run.inputs["messages"][-1]["content"]
        expected = example["expect_access"]
        expected_str = "should allow access" if expected else "should deny access"

        return self.judge.evaluate_strings(
            input=user_input,
            prediction=run.outputs["messages"][-1]["content"],
            reference=expected_str
        )

# Log + evaluate
def main():
    client = Client()
    evaluator = TierAccessJudge()

    for case in test_cases:
        inputs = case["inputs"]
        example = {
            "expect_access": case["expect_access"]
        }

        output = graph.invoke(inputs, config={"configurable": {"user_name": "LangSmith"}})

        run = client.create_run_from_state(
            name=case["name"],
            inputs=inputs,
            outputs=output,
            project_name=PROJECT_NAME,
            reference_output="should allow access" if case["expect_access"] else "should deny access",
        )

        eval_result: EvaluationResult = evaluator.evaluate_run(run, example)
        client.log_evaluation(run_id=run.id, evaluation=eval_result)

        print(f"{case['name']}: {eval_result.score} — {eval_result.comment}")

if __name__ == "__main__":
    main()