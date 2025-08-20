# Friction Log - LangGraph + LangSmith Integration

### 1. **Prebuilt AgentState Field Limitations**
- **Issue**: Custom fields like `customer_tier` and `topic` were silently dropped between nodes when using prebuilt `AgentState`
- **Impact**: Values computed in early nodes (like `classify_topic`) weren't available to later nodes, causing fallback behavior
- **Root cause**: Prebuilt `AgentState` only supports `messages` and `remaining_steps` fields
- **Solution in docs**: LangGraph documentation shows how to "Define a custom state schema that extends AgentState" with examples of adding custom fields and using `state_schema=CustomState` parameter (https://langchain-ai.github.io/langgraph/agents/context/)
- **What I'd do differently**: Start with custom state schema from the beginning instead of assuming prebuilt was sufficient

### 2. **Default LLM Dependencies in Evaluators**
- **Issue**: `load_evaluator("labeled_criteria", ...)` automatically tries to use OpenAI's GPT-4 even when main agent uses Groq
- **Impact**: Required OpenAI API key for evaluation despite never intending to use OpenAI
- **Solution in docs**: LangChain criteria evaluation docs show explicit `llm=` parameter usage: "If you don't specify an eval LLM, the load_evaluator method will initialize a gpt-4 LLM" with examples using `llm=ChatAnthropic(temperature=0)` (https://python.langchain.com/v0.1/docs/guides/productionization/evaluation/string/criteria_eval_chain/)
- **What I'd do differently**: Always check evaluator LLM requirements before implementation

### 3. **Criteria Format Requirements**
- **Issue**: `load_evaluator(..., criteria=["correctness"])` expects dict format, not list
- **Impact**: Cryptic error: `ValueError: dictionary update sequence element #0 has length 11; 2 is required`
- **Solution in docs**: Documentation consistently shows dict format: `criteria={"correctness": "Was the access decision correct?"}` in multiple examples (https://python.langchain.com/v0.1/docs/guides/productionization/evaluation/string/criteria_eval_chain/, https://python.langchain.com/api_reference/langchain/evaluation/langchain.evaluation.criteria.eval_chain.CriteriaEvalChain.html)
- **What I'd do differently**: More carefully read parameter format requirements instead of assuming list format

### 4. **Missing Required Parameters**
- **Issue**: `client.create_run()` requires `run_type` parameter but examples sometimes omit it
- **Impact**: `TypeError: Client.create_run() missing 1 required positional argument: 'run_type'`
- **Solution in docs**: LangSmith documentation shows `run_type` as required with valid values: "llm", "chain", "tool", "retriever", "embedding", "prompt", "parser" (https://pypi.org/project/langsmith/)
- **What I'd do differently**: Check method signatures more carefully when copying example code

### 5. **Confusing but Harmless Warnings**
- **Issue**: Constant `Task X wrote to unknown channel remaining_steps, ignoring it` warnings cluttered logs
- **Impact**: Made it harder to spot real issues during debugging
- **Root cause**: Prebuilt agent executor manages `remaining_steps` internally but warnings aren't explained
- **Documentation gap**: Could not find explanation of this specific warning in LangGraph docs
- **What I'd do differently**: Filter or suppress known harmless warnings during development

## Meta-Learning

**Key insight**: Most friction came from **discovery and user experience gaps** rather than missing documentation. The information existed but required piecing together examples from multiple sections, and error messages didn't clearly point to solutions.

**Better approach**: Start by reading state management and evaluation patterns thoroughly before building, rather than learning them reactively when things break.