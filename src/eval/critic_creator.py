"""
Critic Creator Logic.
This experimental module focuses on generating specialized 'critic' prompts or dynamically
creating specialized evaluation agents to check for specific flaws (e.g., hallucination, tone).

Functions:
- generate_critic_prompt(flaw_type): Returns a system prompt tailored for detecting a specific flaw.
- critique_response(response, critic_prompt): Applies the critic to a generated answer.
"""
