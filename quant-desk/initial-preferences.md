# Initial Preferences

## About me
I am currently based in Taiwan. I have social security number in the US, so I can access Kalshi and other US-based services. I have both US and Taiwan bank accounts, as well as a metamask wallet.

## Goal
Inspired by the Twitter posts attached, I want to build an agentic quant desk where multiple LLMs collaborate under a Python harness. The team is composed of:
- An advisor team composed of several advisor agents, each of which researches a specific market and identifies signals.
- 1 orchestrator: pulls advisor outputs, applies portfolio-level logic (position sizing, correlation/exposure caps, conflicting signals), decides what actually gets acted on, and logs everything. 
- 1 executor: the only component with write access — places/cancels orders 
- 1 risk manager: evaluates risk and holds the power to make final decisions

## Task
Write a detailed plan to implement this. Your report should include the data pipeline, which markets to trade, the design for the Python harness, high-level prompts for each team, high-level logic to evaluate risk and place/exit orders, how to run this loop (what infrastructure, cost, etc.), and other information you consider important.