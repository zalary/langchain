test_cases = [
    {
        "inputs": {
            "messages": [{"role": "user", "content": "How do I enable tracing?"}],
            "customer_tier": "Pro"
        },
        "expected_behavior": "Should return helpful doc-based answer",
        "expect_access": True
    },
    {
        "inputs": {
            "messages": [{"role": "user", "content": "How do I enable tracing?"}],
            "customer_tier": "Free"
        },
        "expected_behavior": "Should politely deny access and suggest upgrade",
        "expect_access": False
    },
    {
        "inputs": {
            "messages": [{"role": "user", "content": "How do I self-host LangSmith?"}],
            "customer_tier": "Enterprise"
        },
        "expected_behavior": "Should return enterprise doc snippet",
        "expect_access": True
    },
    {
        "inputs": {
            "messages": [{"role": "user", "content": "How do I self-host LangSmith?"}],
            "customer_tier": "Pro"
        },
        "expected_behavior": "Should deny access; self-hosting is Enterprise-only",
        "expect_access": False
    },
]